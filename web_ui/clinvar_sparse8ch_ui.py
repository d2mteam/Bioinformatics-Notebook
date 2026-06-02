from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, render_template_string, request
from pyfaidx import Fasta
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(
    os.environ.get(
        "CLINVAR_MODEL_PATH",
        PROJECT_DIR
        / "models"
        / "clinvar_sparse8ch_601_fixed_refseq_temporal_dilated_rcaug_rctta_imbalance_weighted_sampler_with_pretrain_two_snapshot_temporal_pytorch.pt",
    )
)
FASTA_PATH = Path(
    os.environ.get(
        "GRCH38_FASTA_PATH",
        PROJECT_DIR
        / "Data"
        / "ncbi_dataset"
        / "ncbi_dataset"
        / "data"
        / "GCF_000001405.26"
        / "GCF_000001405.26_GRCh38_genomic.fna",
    )
)

LENGTH = 601
FLANK = LENGTH // 2
BASES = "ACGT"
BASE_TO_INDEX = {base: idx for idx, base in enumerate(BASES)}
COMPLEMENT_BASE = str.maketrans("ACGT", "TGCA")
COMPLEMENT_CHANNEL = np.array([3, 2, 1, 0], dtype=np.int64)

CHROMOSOME_TO_REFSEQ = {
    "1": "NC_000001.11",
    "2": "NC_000002.12",
    "3": "NC_000003.12",
    "4": "NC_000004.12",
    "5": "NC_000005.10",
    "6": "NC_000006.12",
    "7": "NC_000007.14",
    "8": "NC_000008.11",
    "9": "NC_000009.12",
    "10": "NC_000010.11",
    "11": "NC_000011.10",
    "12": "NC_000012.12",
    "13": "NC_000013.11",
    "14": "NC_000014.9",
    "15": "NC_000015.10",
    "16": "NC_000016.10",
    "17": "NC_000017.11",
    "18": "NC_000018.10",
    "19": "NC_000019.10",
    "20": "NC_000020.11",
    "21": "NC_000021.9",
    "22": "NC_000022.11",
    "X": "NC_000023.11",
    "Y": "NC_000024.10",
    "M": "NC_012920.1",
    "MT": "NC_012920.1",
}


class DilatedClinVarCNN(nn.Module):
    def __init__(self, in_channels: int = 8):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        ]
        for dilation in [1, 2, 4, 8, 16, 32, 64]:
            layers.extend(
                [
                    nn.Conv1d(64, 64, kernel_size=7, padding=3 * dilation, dilation=dilation),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.05),
                ]
            )
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(64 * 3, 96),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(96, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        center = features[:, :, features.shape[-1] // 2]
        global_max = torch.amax(features, dim=-1)
        global_mean = torch.mean(features, dim=-1)
        pooled = torch.cat([center, global_max, global_mean], dim=1)
        return self.classifier(pooled).squeeze(1)


def normalize_chromosome(chromosome: str) -> str:
    chrom = str(chromosome).strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    if chrom.upper() in {"M", "MT"}:
        return "MT"
    return chrom.upper()


def resolve_fasta_name(chromosome: str, fasta_names: set[str]) -> str:
    chrom = normalize_chromosome(chromosome)
    candidates = [chromosome.strip(), chrom, f"chr{chrom}", CHROMOSOME_TO_REFSEQ.get(chrom, "")]
    for candidate in candidates:
        if candidate and candidate in fasta_names:
            return candidate
    raise ValueError(f"Không tìm thấy chromosome/accession '{chromosome}' trong FASTA.")


def parse_variant_text(text: str) -> tuple[str, int, str, str]:
    cleaned = text.strip().replace(" ", "")
    for sep in [":", "-"]:
        cleaned = cleaned.replace(sep, ":")
    if ">" in cleaned and cleaned.count(":") >= 2:
        chrom, pos, change = cleaned.split(":", 2)
        ref, alt = change.split(">", 1)
        return chrom, int(pos), ref.upper(), alt.upper()
    raise ValueError("Định dạng nhanh hợp lệ: chr11:126275389:C>T")


def reverse_complement_8ch(arr: np.ndarray) -> np.ndarray:
    out = arr[::-1].copy()
    ref = out[:, :4].copy()
    alt = out[:, 4:8].copy()
    out[:, :4] = ref[:, COMPLEMENT_CHANNEL]
    out[:, 4:8] = alt[:, COMPLEMENT_CHANNEL]
    return out


def make_sparse_alt_array(sequence: str, alt: str) -> np.ndarray:
    arr = np.zeros((LENGTH, 8), dtype=np.float32)
    for pos, base in enumerate(sequence):
        arr[pos, BASE_TO_INDEX[base]] = 1.0
    arr[FLANK, 4 + BASE_TO_INDEX[alt]] = 1.0
    return arr


def load_checkpoint() -> tuple[DilatedClinVarCNN, dict, torch.device]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    model = DilatedClinVarCNN(in_channels=8)
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, config, device


def validate_snv(ref: str, alt: str) -> tuple[str, str]:
    ref = ref.strip().upper()
    alt = alt.strip().upper()
    if ref not in BASE_TO_INDEX or alt not in BASE_TO_INDEX:
        raise ValueError("REF và ALT phải là SNV một base: A/C/G/T.")
    if ref == alt:
        raise ValueError("REF và ALT đang giống nhau; đây không phải biến thể SNV.")
    return ref, alt


def predict_variant(chromosome: str, position: int, ref: str, alt: str, use_rc_tta: bool = True) -> dict:
    ref, alt = validate_snv(ref, alt)
    if position < 1:
        raise ValueError("Position phải là tọa độ 1-based dương.")
    fasta_name = resolve_fasta_name(chromosome, FASTA_NAMES)
    start_1based = position - FLANK
    end_1based = position + FLANK
    if start_1based < 1:
        raise ValueError(f"Biến thể quá gần đầu chromosome, không lấy đủ {LENGTH} bp.")
    sequence = str(GENOME[fasta_name][start_1based - 1 : end_1based]).upper()
    if len(sequence) != LENGTH:
        raise ValueError(f"Không lấy đủ {LENGTH} bp từ FASTA.")
    if any(base not in BASE_TO_INDEX for base in sequence):
        raise ValueError("Context chứa base ngoài A/C/G/T, model hiện không hỗ trợ.")
    fasta_ref = sequence[FLANK]
    if fasta_ref != ref:
        raise ValueError(f"REF không khớp GRCh38 FASTA tại center: input REF={ref}, FASTA={fasta_ref}.")

    arr = make_sparse_alt_array(sequence, alt)
    tensors = [torch.from_numpy(arr.T[None, :, :].copy())]
    if use_rc_tta:
        rc_arr = reverse_complement_8ch(arr)
        tensors.append(torch.from_numpy(rc_arr.T[None, :, :].copy()))
    with torch.no_grad():
        probs = []
        logits = []
        for tensor in tensors:
            logit = MODEL(tensor.to(DEVICE))
            prob = torch.sigmoid(logit)
            logits.append(float(logit.detach().cpu().item()))
            probs.append(float(prob.detach().cpu().item()))
    probability = float(np.mean(probs))
    logit_mean = float(np.mean(logits))
    threshold = float(CONFIG.get("best_val_f1_threshold", {}).get("threshold", 0.5))
    return {
        "chromosome": normalize_chromosome(chromosome),
        "chromosome_fasta": fasta_name,
        "position": int(position),
        "ref": ref,
        "alt": alt,
        "context_start_1based": int(start_1based),
        "context_end_1based": int(end_1based),
        "probability_pathogenic": probability,
        "logit": logit_mean,
        "threshold": threshold,
        "prediction": "Pathogenic/Likely pathogenic" if probability >= threshold else "Benign/Likely benign",
        "rc_tta": bool(use_rc_tta),
        "model_name": CONFIG.get("model_name", MODEL_PATH.name),
        "test_pr_auc": CONFIG.get("test_metrics", {}).get("pr_auc"),
        "test_roc_auc": CONFIG.get("test_metrics", {}).get("roc_auc"),
    }


MODEL, CONFIG, DEVICE = load_checkpoint()
GENOME = Fasta(str(FASTA_PATH), as_raw=True, sequence_always_upper=True, rebuild=False)
FASTA_NAMES = set(GENOME.keys())

app = Flask(__name__)


PAGE = """
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ClinVar SNV CNN Predictor</title>
  <style>
    :root { color-scheme: light; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    body { margin: 0; background: #f6f7f9; color: #16181d; }
    main { max-width: 980px; margin: 0 auto; padding: 32px 20px 48px; }
    h1 { margin: 0 0 8px; font-size: 30px; }
    .sub { color: #5e6573; margin: 0 0 24px; }
    .panel { background: #fff; border: 1px solid #d8dde6; border-radius: 8px; padding: 20px; margin-bottom: 18px; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 14px; }
    label { display: grid; gap: 6px; font-size: 13px; color: #404653; font-weight: 650; }
    input, select { font: inherit; padding: 10px 12px; border: 1px solid #c6ccd8; border-radius: 6px; background: #fff; }
    .wide { grid-column: 1 / -1; }
    button { margin-top: 16px; padding: 11px 16px; border: 0; border-radius: 6px; background: #2357d9; color: white; font-weight: 700; cursor: pointer; }
    button:disabled { opacity: 0.55; cursor: progress; }
    .result { display: none; }
    .result strong { font-size: 28px; }
    .bad { color: #b42318; }
    .good { color: #087443; }
    .meta { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-top: 16px; }
    .kv { background: #f8fafc; border: 1px solid #e2e7ef; border-radius: 6px; padding: 10px; }
    .kv span { display: block; color: #687386; font-size: 12px; }
    .kv b { display: block; margin-top: 4px; word-break: break-word; }
    .err { display: none; color: #b42318; background: #fff1f0; border: 1px solid #f4b4ae; border-radius: 6px; padding: 12px; }
    code { background: #eef1f6; padding: 2px 5px; border-radius: 4px; }
    @media (max-width: 720px) { .grid, .meta { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<main>
  <h1>ClinVar SNV CNN Predictor</h1>
  <p class="sub">Model sparse-alt 8ch 601bp. Nhập biến thể dạng GRCh38, SNV một base.</p>

  <section class="panel">
    <label class="wide">Nhập nhanh
      <input id="variantText" placeholder="Ví dụ: chr11:126275389:C>T">
    </label>
    <div class="grid">
      <label>Chromosome <input id="chromosome" value="11"></label>
      <label>Position VCF <input id="position" type="number" value="126275389"></label>
      <label>REF <input id="ref" maxlength="1" value="C"></label>
      <label>ALT <input id="alt" maxlength="1" value="T"></label>
    </div>
    <label style="margin-top:14px; display:flex; align-items:center; gap:8px;">
      <input id="rcTta" type="checkbox" checked style="width:auto;"> Dùng reverse-complement TTA
    </label>
    <button id="predictBtn">Dự đoán</button>
  </section>

  <section id="error" class="err"></section>

  <section id="result" class="panel result">
    <div>Xác suất pathogenic</div>
    <strong id="probability"></strong>
    <div id="label" style="margin-top:8px; font-weight:700;"></div>
    <div class="meta" id="meta"></div>
  </section>

</main>
<script>
const $ = (id) => document.getElementById(id);

function parseQuick(text) {
  const match = text.trim().replaceAll(" ", "").match(/^(.+?)[:\\-](\\d+)[:\\-]([ACGTacgt])>([ACGTacgt])$/);
  if (!match) return false;
  $("chromosome").value = match[1];
  $("position").value = match[2];
  $("ref").value = match[3].toUpperCase();
  $("alt").value = match[4].toUpperCase();
  return true;
}

$("variantText").addEventListener("change", () => {
  const text = $("variantText").value;
  if (text.trim()) parseQuick(text);
});

$("predictBtn").addEventListener("click", async () => {
  const err = $("error");
  const result = $("result");
  err.style.display = "none";
  result.style.display = "none";
  if ($("variantText").value.trim()) parseQuick($("variantText").value);
  const payload = {
    chromosome: $("chromosome").value,
    position: Number($("position").value),
    ref: $("ref").value,
    alt: $("alt").value,
    rc_tta: $("rcTta").checked,
  };
  $("predictBtn").disabled = true;
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed");
    $("probability").textContent = (data.probability_pathogenic * 100).toFixed(2) + "%";
    $("label").textContent = data.prediction;
    $("label").className = data.probability_pathogenic >= data.threshold ? "bad" : "good";
    const fields = [
      ["Threshold", data.threshold.toFixed(4)],
      ["Logit", data.logit.toFixed(4)],
      ["Chromosome FASTA", data.chromosome_fasta],
      ["Context", data.context_start_1based + " - " + data.context_end_1based],
      ["RC TTA", data.rc_tta ? "Có" : "Không"],
      ["Model", data.model_name],
      ["Test PR-AUC", data.test_pr_auc == null ? "n/a" : Number(data.test_pr_auc).toFixed(4)],
      ["Test ROC-AUC", data.test_roc_auc == null ? "n/a" : Number(data.test_roc_auc).toFixed(4)],
    ];
    $("meta").innerHTML = fields.map(([k, v]) => `<div class="kv"><span>${k}</span><b>${v}</b></div>`).join("");
    result.style.display = "block";
  } catch (e) {
    err.textContent = e.message;
    err.style.display = "block";
  } finally {
    $("predictBtn").disabled = false;
  }
});
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.post("/api/predict")
def api_predict():
    try:
        payload = request.get_json(force=True) or {}
        if payload.get("variant_text"):
            chromosome, position, ref, alt = parse_variant_text(payload["variant_text"])
        else:
            chromosome = str(payload.get("chromosome", "")).strip()
            position = int(payload.get("position"))
            ref = str(payload.get("ref", "")).strip()
            alt = str(payload.get("alt", "")).strip()
        result = predict_variant(
            chromosome=chromosome,
            position=position,
            ref=ref,
            alt=alt,
            use_rc_tta=bool(payload.get("rc_tta", True)),
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    app.run(host="127.0.0.1", port=port, debug=False)
