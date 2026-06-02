# ClinVar pathogenicity prediction

## Dự án đang làm gì

Dự án này đang xây một pipeline machine learning cho bài toán phân loại biến thể ClinVar dạng SNV thành 2 nhóm:

- `1`: `Pathogenic` / `Likely pathogenic`
- `0`: `Benign` / `Likely benign`

Nguồn dữ liệu chính là `Data/variant_summary.txt` từ ClinVar snapshot đầu tháng 5/2026 và FASTA tham chiếu GRCh38 trong `Data/ncbi_dataset/`. Pipeline hiện có 2 hướng mô hình:

- Sequence models: dùng DNA context quanh vị trí đột biến, encode thành tensor cho CNN. Mốc nghiêm ngặt tốt nhất hiện tại là 601 bp với full input `REF+ALT+marker+positional`, dilated CNN, reverse-complement augment/TTA, genome-block split, coordinate purge và exact sequence purge.
- Tabular models: dùng metadata ClinVar và annotation ngoài tùy chọn cho XGBoost.

Artifact hiện tại cho thấy pipeline đã chạy xong preprocessing, đã train nhiều model baseline, đã audit leakage của random/gene split, đã sửa mapping FASTA RefSeq accession `.11/.12`, và đã thử các hướng compact input, SE/gating, FiLM/gating, Kaggle standalone, explainability và error analysis. Model sequence tốt nhất theo split nghiêm ngặt hiện tại là full 9-channel 601 bp dilated CNN, còn notebook compact 4-channel là hướng gọn hơn để giảm input thừa và dễ giải thích.

## Bảng metric các mốc CNN chính

| Mốc | Cải tiến chính | Dataset / split | PR-AUC | ROC-AUC | F1 | Precision | Recall | Accuracy | Threshold |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| CNN alt-only 201 bp | CNN 1D baseline | 460k / random row | 0.2454 | 0.6526 | 0.2774 | 0.1772 | 0.6389 | 0.5705 | 0.5 |
| CNN REF+ALT+marker 201 bp | Early fusion 9ch | 460k / random row | 0.5224 | 0.8482 | 0.4746 | 0.3532 | 0.7233 | 0.7933 | 0.5 |
| CNN + positional encoding | + sinusoidal PE | 460k / gene_group | 0.5722 | 0.8806 | 0.5529 | 0.5272 | 0.5811 | 0.8835 | 0.5 |
| CNN 1001 bp + PE | Context dài hơn | 460k / gene_group | 0.5702 | 0.8584 | 0.5064 | 0.3775 | 0.7691 | 0.7834 | 0.5 |
| Dilated CNN 601 bp | Tích chập thưa | 460k / gene_group | 0.7745 | 0.9327 | 0.6933 | 0.6693 | 0.7191 | 0.9081 | 0.5 |
| Dilated CNN 1001 bp | Tích chập thưa + 1001 bp | 460k / gene_group | 0.7732 | 0.9317 | 0.6866 | 0.6491 | 0.7287 | 0.9039 | 0.5 |
| Dilated CNN 601 bp + RC | + RC augment/TTA | 460k / gene_group | 0.8431 | 0.9557 | 0.7366 | 0.6693 | 0.8188 | 0.9154 | 0.5 |
| Dilated residual CNN 601 bp + RC | + residual/skip | 460k / gene_group | 0.8224 | 0.9493 | 0.7335 | 0.7197 | 0.7479 | 0.9215 | 0.5 |
| Dilated CNN 601 bp + relative PE + RC | + relative PE | 460k / gene_group | 0.8381 | 0.9533 | 0.7362 | 0.6733 | 0.8121 | 0.9159 | 0.5 |
| Dilated window attention 601 bp + RC | + 1D window attention | 460k / gene_group | 0.8507 | 0.9589 | 0.7410 | 0.6605 | 0.8440 | 0.9148 | 0.5 |
| Full 9ch dilated CNN 601 bp | fixed_refseq + strict split | 1.426M / genome_block + purge | **0.9124** | **0.9839** | **0.8178** | **0.7920** | **0.8454** | **0.9582** | val-F1 |
| Compact REF + center REF/ALT | 4ch late fusion | 1.426M / genome_block + purge | 0.8870 | 0.9782 | 0.7946 | 0.7767 | 0.8133 | 0.9533 | val-F1 |
| Compact SE/gating | + SE/gate | 1.426M / genome_block + purge | 0.8858 | 0.9781 | 0.7940 | 0.7780 | 0.8107 | n/a | val-F1 |
| Compact FiLM blocks | + FiLM all blocks | 1.426M / genome_block + purge | 0.8874 | 0.9780 | 0.7943 | 0.7762 | 0.8133 | n/a | val-F1 |
| Sparse ALT 8ch multibranch | ALT-at-center + multibranch CNN | 1.426M / genome_block + purge | 0.9043 | 0.9817 | 0.8099 | 0.7829 | 0.8388 | 0.9563 | val-F1 |
| Full 9ch dilated CNN 601 bp | chromosome holdout | 1.426M / chromosome split | 0.8877 | 0.9755 | 0.7899 | 0.8244 | 0.7582 | 0.9462 | val-F1 |
| Full 9ch temporal space-only matched | pseudo-temporal diagnostic | 1.300M / space_only_matched | 0.8423 | 0.9682 | 0.7535 | 0.7594 | 0.7477 | 0.9459 | val-F1 |
| Full 9ch temporal time-only | pseudo-temporal diagnostic | 1.300M / time_only | 0.5182 | 0.9023 | 0.5259 | 0.5187 | 0.5333 | 0.9639 | val-F1 |
| Full 9ch two-snapshot temporal | true prospective split | 2022-12 -> 2026-05 | đang chạy | đang chạy | đang chạy | đang chạy | đang chạy | đang chạy | val-F1 |

## Dữ liệu và label hiện tại

Pipeline mới sau khi sửa mapping FASTA theo RefSeq accession version (`NC_000001.11`, `NC_000002.12`, ...) tạo ra dataset `fixed_refseq` lớn hơn đáng kể:

- `processed/clinvar_filtered_step8_fixed_refseq.parquet`: `1,426,477` SNV sau filter label/review/GRCh38/REF match.
- `processed/clinvar_training_metadata_601_fixed_refseq.parquet`: `1,426,427` biến thể có context 601 bp.
- `processed/X_ref_alt_marker_601_fixed_refseq.npy`: shape `(1426427, 601, 9)`, dense `uint8`, khoảng `7.19 GB`.
- `processed/y_601_fixed_refseq.npy`: shape `(1426427,)`.
- `processed/compact_ref_alt_center_metadata_601_fixed_refseq.parquet`: metadata cho compact model.
- `processed/X_ref_index_601_fixed_refseq.npy`: shape `(1426427, 601)`, chỉ lưu REF base index, khoảng `0.80 GB`.
- `processed/y_compact_ref_alt_center_601_fixed_refseq.npy`: shape `(1426427,)`.

Phân bố label trong dataset `fixed_refseq` 601 bp:

| Label | Ý nghĩa | Số dòng |
|---|---|---:|
| `0` | Benign / Likely benign | 1,257,967 |
| `1` | Pathogenic / Likely pathogenic | 168,460 |

Tỉ lệ pathogenic khoảng `11.8%`, nên PR-AUC và precision/recall của class pathogenic quan trọng hơn accuracy.

Artifact 460k cũ vẫn được giữ để tái lập các thí nghiệm ban đầu. Con số 460k xuất phát từ mapping FASTA hẹp hơn trước khi sửa accession version:

- `processed/clinvar_training_metadata.parquet`: `460,488` biến thể, `15` cột metadata.
- `processed/X_alt_201.npy`: shape `(460488, 201, 4)`.
- `processed/X_ref_alt_marker_201.npy`: shape `(460488, 201, 9)`.
- `processed/X_ref_alt_marker_601.npy`: shape `(460443, 601, 9)`, khoảng `2.4 GB`.
- `processed/X_ref_alt_marker_1001.npy`: shape `(460414, 1001, 9)`, khoảng `3.9 GB`.
- `processed/X_ref_alt_marker_2001.npy`: shape `(460224, 2001, 9)`, khoảng `7.8 GB`.

Phân bố label trong dataset sequence cũ:

| Label | Ý nghĩa | Số dòng |
|---|---|---:|
| `0` | Benign / Likely benign | 401,054 |
| `1` | Pathogenic / Likely pathogenic | 59,434 |

Tỉ lệ pathogenic khoảng `12.9%`.

## Pipeline notebook

### `notebooks/01_read_clinvar_variant_summary.ipynb`

Đọc `Data/variant_summary.txt`, filter và tạo dataset sequence.

Các bước chính:

- Chỉ giữ label rõ ràng: benign/likely benign/pathogenic/likely pathogenic.
- Chỉ giữ `Assembly == GRCh38`.
- Chỉ giữ `Type == single nucleotide variant`.
- Filter `ReviewStatus` đáng tin cậy.
- Loại dòng thiếu chromosome, position, REF/ALT.
- Chuẩn hóa chromosome để khớp FASTA.
- Kiểm tra REF allele với FASTA GRCh38.
- Lấy DNA context 201 bp quanh mutation.
- Tạo `ref_seq`, `alt_seq`.
- One-hot encode thành tensor 4 kênh hoặc 9 kênh.

Output chính:

- `processed/clinvar_filtered_step8.parquet`: `460,492` dòng sau filter và check REF.
- `processed/clinvar_context_201.parquet`: `460,491` dòng có context 201 bp.
- `processed/clinvar_training_metadata.parquet`
- `processed/X_alt_201.npy`
- `processed/X_ref_alt_marker_201.npy`
- `processed/y.npy`
- `processed/clinvar_cnn_dataset_alt_201.npz`
- `processed/clinvar_cnn_dataset_ref_alt_marker_201.npz`

### `notebooks/02_cnn_demo.ipynb`

Train CNN 1D PyTorch trên random stratified split.

Input tốt nhất trong notebook này là 9-channel:

- `0:4`: one-hot `ref_seq`
- `4:8`: one-hot `alt_seq`
- `8`: mutation marker tại center index `100`

Model 9-channel cải thiện mạnh so với bản chỉ dùng `alt_seq`.

### `notebooks/02_1_model_experiments.ipynb`

Notebook thử nghiệm model zoo nhỏ trên cùng dataset 9-channel:

- `cnn_baseline`
- `cnn_residual`
- `transformer_small`

Artifact hiện tại lưu model `transformer_small`.

### `notebooks/02_2_cnn_split_leakage_audit.ipynb`

Audit random split của CNN demo.

Kết quả audit từ artifact hiện tại:

- Exact variant overlap giữa train/val/test: `0`.
- Gene overlap train/test rất cao: `3,709 / 3,920` gene test, khoảng `94.6%`.
- Khoảng `95.3%` biến thể test nằm cách biến thể train gần nhất trong vòng `201 bp`.
- Exact `ref_seq` overlap train/test khoảng `11.6%` unique sequence test.

Kết luận: random split không duplicate exact variant, nhưng vẫn optimistic vì cùng gene/vùng genome xuất hiện ở cả train và test. Nên xem `gene_group` split là đánh giá nghiêm túc hơn.

### `notebooks/02_3_cnn_gene_group_split.ipynb`

Train CNN với `GroupShuffleSplit` theo `GeneSymbol`, không để cùng gene xuất hiện ở train/val/test.

Notebook này cũng có:

- cấu hình xử lý class imbalance theo một cơ chế tại một thời điểm
- mặc định mới tránh dùng đồng thời weighted sampler và `pos_weight`
- hard-negative fine-tuning
- threshold table để chọn precision/recall operating point

Output chính:

- `models/clinvar_ref_alt_marker_cnn_gene_group_pytorch.pt`
- `models/clinvar_ref_alt_marker_cnn_gene_group_balanced_pytorch.pt`
- `processed/cnn_gene_group_*`

### `notebooks/02_4_cnn_positional_encoding.ipynb`

Mở rộng CNN `gene_group` bằng sinusoidal positional encoding.

Input từ `(201, 9)` được concat thêm 8 kênh positional encoding thành `(201, 17)`. Đây là mốc cải tiến quan trọng ban đầu, nhưng không còn là model sequence tốt nhất hiện tại sau các thử nghiệm 601 bp/dilated/window attention.

Output chính:

- `models/clinvar_cnn_gene_group_positional_encoding_pytorch.pt`
- `processed/cnn_gene_group_positional_encoding_test_predictions_pytorch.parquet`
- `processed/cnn_gene_group_positional_encoding_training_history_pytorch.parquet`
- `processed/cnn_gene_group_positional_encoding_threshold_table_pytorch.parquet`

### `scripts/build_cnn_dataset_1001.py` và `scripts/train_cnn_gene_group_positional_1001.py`

Script build context dài quanh SNV, mặc định 1001 bp nhưng có thể dùng `--flank` để tạo 601/2001/4001/8001 nếu đủ storage. Pipeline vẫn dùng input ref+alt+marker 9 kênh và concat thêm positional encoding khi train. Script train hiện hỗ trợ:

- `baseline`: CNN nông giống thử nghiệm ban đầu.
- `dilated`: CNN 1D dùng dilation `[1, 2, 4, 8, 16, 32, 64]`, giữ resolution 1001 bp và dùng center/global pooling.
- `dilated_residual`: biến thể có residual/skip connection.
- `dilated_window_attention`: dilated CNN cộng thêm 1D local window attention trước pooling.

Script train cũng hỗ trợ:

- `--length`: chọn dataset 601/1001 hoặc length đã build.
- `--positional-encoding none|sinusoidal|relative`.
- `--rc-augment`: random reverse-complement khi train.
- `--rc-tta`: average prediction giữa original và reverse-complement khi eval/test.
- `--split-mode gene_group|purged_gene_group`: `purged_gene_group` tách theo gene rồi loại val/test variant quá gần train/selection theo genomic distance.
- `--purge-distance-bp`: khoảng cách purge, mặc định `5000`.
- `--scheduler none|cosine|onecycle`, `--warmup-epochs`, `--min-lr-ratio`: learning-rate schedule theo batch.
- `--grad-clip-norm`: gradient clipping, mặc định `1.0`.
- `--imbalance-strategy none|weighted_sampler|pos_weight|legacy_sqrt_sampler_pos_weight`: chọn một cách xử lý class imbalance. Mặc định mới là `weighted_sampler`, tức dùng inverse-frequency sampler và loss không `pos_weight`; `legacy_sqrt_sampler_pos_weight` chỉ để tái lập artifact cũ.
- `--random-state`: đổi gene-group split; output seed khác 42 được gắn hậu tố `_seedXX` để tránh ghi đè artifact chính.

Kết quả trên benchmark 460k/gene-group cũ cho thấy chỉ tăng window lên 1001 bp bằng CNN baseline chưa cải thiện. Cải tiến mạnh nhất đến từ 601 bp + dilated CNN + RC augment/TTA; 1D window attention tăng thêm trên benchmark cũ nhưng chưa được xem là best cuối cùng trên split nghiêm ngặt `fixed_refseq`.
Benchmark attention 2001 bp vẫn là hướng có thể thử nếu đủ compute, nhưng kết luận chính hiện tại nên dựa trên notebook 10/11 với `genome_block + purge`.

Output chính:

- `processed/clinvar_context_601.parquet`
- `processed/clinvar_training_metadata_601.parquet`
- `processed/X_ref_alt_marker_601.npy`
- `processed/y_601.npy`
- `processed/clinvar_context_1001.parquet`
- `processed/clinvar_training_metadata_1001.parquet`
- `processed/X_ref_alt_marker_1001.npy`
- `processed/y_1001.npy`
- `processed/clinvar_context_2001.parquet`
- `processed/clinvar_training_metadata_2001.parquet`
- `processed/X_ref_alt_marker_2001.npy`
- `processed/y_2001.npy`
- `models/clinvar_cnn_gene_group_positional_encoding_601_dilated_rcaug_rctta_pytorch.pt`
- `models/clinvar_cnn_gene_group_positional_encoding_601_dilated_window_attention_rcaug_rctta_pytorch.pt`
- `models/clinvar_cnn_gene_group_positional_encoding_1001_pytorch.pt`
- `models/clinvar_cnn_gene_group_positional_encoding_1001_dilated_pytorch.pt`
- `processed/cnn_gene_group_positional_encoding_601_dilated_window_attention_rcaug_rctta_*`
- `processed/cnn_gene_group_positional_encoding_2001_dilated_window_attention_rcaug_rctta_purged5000_*`
- `processed/cnn_gene_group_positional_encoding_1001_*`

### `notebooks/08_train_best_cnn_601_window_attention_rcaug_rctta.ipynb`

Notebook đóng gói best model cũ theo gene-group benchmark và cách dùng lại artifact:

- Cấu hình: `601 bp + dilated_window_attention + sinusoidal positional encoding + RC augment + RC TTA`.
- Mặc định `RUN_TRAIN = False` để không vô tình ghi đè artifact.
- Có cell gọi lại script train đúng tham số khi cần train lại.
- Có cell đọc metrics/history, load checkpoint và predict theo dataset index.

Notebook này không chỉnh sửa notebook 01.

### `notebooks/09_kaggle_train_window_attention_from_raw.ipynb`

Notebook standalone để đẩy lên Kaggle. Notebook tự tìm `variant_summary.txt` và `ncbi_dataset.zip`, giải nén FASTA GRCh38, lọc ClinVar SNV từ raw, build tensor `ref+alt+marker`, chạy các cell demo/smoke test, rồi train/evaluate `dilated_window_attention` với `gene_group` và `purged_gene_group`.

### `notebooks/10_train_best_cnn_601_dilated_rcaug_rctta.ipynb`

Notebook project cho benchmark nghiêm ngặt mới trên dataset `fixed_refseq`. Đây là model reference chính để so sánh với các biến thể compact/sparse.

**Cấu hình chính:**

- Input full 9-channel: `REF one-hot 4 + ALT one-hot 4 + marker center 1`.
- Positional encoding sinusoidal 8 kênh, tổng 17 kênh vào CNN.
- `DilatedClinVarCNN`: stem conv7 → 7 dilated conv7 với dilation `[1, 2, 4, 8, 16, 32, 64]` → pooling `center ‖ global_max ‖ global_mean` (64×3) → MLP → logit.
- RC augment khi train (50% xác suất flip) và RC TTA khi eval/test (average forward + rc).
- Split: `genome_block` block 1 Mb, coordinate purge ≥ 5000 bp, exact REF sequence purge (Blake2b hash).
- `imbalance_strategy = weighted_sampler` (inverse-frequency sampler, không dùng `pos_weight`).
- Hard-negative mining 2 epoch sau main training với lr `3e-4`: giữ toàn bộ positives + hard negatives (score ≥ 0.5) + capped easy negatives.
- Mặc định `RUN_TRAIN = True`, `FORCE_OVERWRITE = True`. Đổi `RUN_TRAIN = False` để chỉ đọc metrics/inference.

**Training history artifact (seed 42, genome_block split):**

| Phase | Epoch | Val PR-AUC | Val ROC-AUC |
|---|---:|---:|---:|
| main | 1 | 0.7819 | 0.9475 |
| main | 5 | 0.8943 | 0.9790 |
| hard_negative | 1 | 0.9023 | 0.9801 |
| hard_negative | 2 | **0.9053** | **0.9809** |

**Kết quả test nghiêm ngặt seed 42:**

- Test PR-AUC `0.9124`.
- Test ROC-AUC `0.9839`.
- Test F1 tại threshold chọn từ validation `0.8178`.
- Confusion matrix tại threshold validation-F1: `[[186995, 5335], [3713, 20308]]`.

**Cells đáng chú ý:**

- Cell 4 (`Split strict`): build genome-block groups, purge coordinate + sequence, audit overlap.
- Cell 8 (`Train/evaluate inline`): toàn bộ train loop, hard-negative mining, lưu metrics/predictions/split_indices.
- Cell 10 (`Metrics`): đọc metrics JSON, in summary; dùng `HISTORICAL_METRICS` fallback nếu chưa chạy.
- Cell 11 (`Load checkpoint`): load artifact để inference mà không cần chạy lại train.
- Cell 12 (`Error analysis`): sort predictions theo `abs_error`, xem gene/ClinicalSignificance/REF/ALT của các case sai nhất.

Đây là mốc metric sequence tốt nhất hiện tại theo split nghiêm ngặt.

### `notebooks/11_train_compact_ref_alt_center_cnn_601.ipynb`

Notebook compact input để kiểm tra giả thuyết input 9-channel đang dư thừa. Pipeline tự xử lý dữ liệu từ raw ClinVar + FASTA (không dùng dense tensor `X_ref_alt_marker`).

**Thiết kế:**

- Lưu compact cache `X_ref_index_601_fixed_refseq.npy` shape `(N, 601)` dtype `uint8` (base index 0-3), khoảng `0.80 GB` so với `7.19 GB` dense.
- `Dataset` encode one-hot on-the-fly từ `ref_index` + `alt_base` từ metadata.
- CNN nhận `REF 4-channel sequence` (601 bp). Center REF/ALT vector 8 chiều đưa vào classifier (late fusion).
- Bỏ marker channel và positional encoding mặc định (`POSITIONAL_ENCODING = "none"`).
- `MODEL_VARIANT`:
  - `"compact"`: CNN backbone + concatenate center_vector vào feature trước classifier.
  - `"variant_gated_se"`: thêm `VariantConditionedSEGate` — MLP(center_vector) → tanh gate per-channel; weight zero-init để training bắt đầu giống compact baseline; `VARIANT_GATE_PLACEMENT = "stem_late"`.
- `LABEL_MODE = "all"` (giữ Likely) hoặc `"definitive_only"` (chỉ Benign/Pathogenic thuần).
- Cùng split logic với nb10: genome_block + purge 5000 bp + exact_ref sequence purge.
- Cell 17 so sánh 3 model: full 9-channel (nb10), compact no-gate, compact SE-gate.
- Cell 19 attribution heatmap (gradient saliency theo vị trí, phân rã center vs edges).
- Cell 20/21 error analysis top wrong cases, phân tích theo gene và ReviewStatus.

**Kết quả strict split (seed 42):**

- Test PR-AUC `0.8870`.
- Test ROC-AUC `0.9782`.
- Test F1 tại threshold chọn từ validation `0.7946`.
- Confusion matrix: `[[186714, 5616], [4485, 19536]]`.

So với nb10: giảm `0.025` PR-AUC, tăng false negative `3713 → 4485`. Nguyên nhân: compact là late fusion — CNN không thấy ALT/marker trong convolution đầu.

So với full 9-channel, compact giảm khoảng `0.025` PR-AUC và tăng false negative. Điều này cho thấy 9-channel early fusion vẫn giúp CNN nhìn trực tiếp `REF/ALT/marker` tại center từ các convolution đầu.

### `notebooks/11_kaggle_train_compact_ref_alt_center_cnn_601.ipynb`

Bản Kaggle standalone của notebook 11.

Ràng buộc path:

- Input chỉ đọc từ `/kaggle/input`.
- Output/cache/model chỉ ghi vào `/kaggle/working`.
- Không ghi vào `/kaggle/input`.
- Không tự unzip dataset vào working; dùng FASTA đã được Kaggle extract hoặc set override path trong `/kaggle/input`.

Các cờ tăng tốc Kaggle:

- `BATCH_SIZE = 512` mặc định trên Kaggle.
- `NUM_WORKERS = 4`, `PREFETCH_FACTOR = 4`, `PERSISTENT_WORKERS = True`.
- `USE_AMP = True`.
- `TRAIN_METRICS_MODE = "loss_only"`.
- `EVAL_RC_TTA_EVERY_EPOCH = False`, final/test vẫn có thể dùng RC TTA.

### `notebooks/12_kaggle_train_compact_ref_alt_center_film_cnn_601.ipynb`

Bản Kaggle standalone thử nghiệm FiLM/gating mạnh hơn SE-gate của nb11. Cùng split logic và dataset, khác ở cơ chế đưa thông tin biến thể vào CNN.

**Thiết kế:**

- Vẫn dùng compact input: REF sequence 601 bp (4ch) + center REF/ALT vector (8dim).
- `MODEL_VARIANT`:
  - `"compact"`: baseline không gate.
  - `"variant_gated_se"`: SE-gate nhẹ (giống nb11, giữ để so sánh).
  - `"film_blocks"`: `VariantFiLMModulator` — MLP 3-layer sinh `gamma` và `beta` riêng biệt (channels×2 output), áp dụng feature-wise linear modulation tại **mỗi** dilated block: `y = x*(1 + scale*gamma) + scale*beta`, `scale=0.5`. Last layer zero-init.
- `VARIANT_GATE_PLACEMENT = "all"` cho `film_blocks` (tất cả 7 dilated blocks nhận modulation).
- Kaggle path logic: `IS_KAGGLE` flag auto-detect, input từ `/kaggle/input`, output vào `/kaggle/working`.
- Cờ tăng tốc Kaggle: `BATCH_SIZE=512`, `NUM_WORKERS=4`, `USE_AMP=True`, `PREFETCH_FACTOR=4`, `PERSISTENT_WORKERS=True`, `TRAIN_METRICS_MODE="loss_only"`, `EVAL_RC_TTA_EVERY_EPOCH=False`.

**Kết quả Kaggle (seed 42):**

| MODEL_VARIANT | Test PR-AUC | Test ROC-AUC | F1 @ val threshold |
|---|---:|---:|---:|
| `film_blocks` | 0.8874 | 0.9780 | 0.7943 |
| `variant_gated_se` (nb11) | 0.8858 | 0.9781 | 0.7940 |

Kết luận: FiLM mạnh hơn về kiến trúc nhưng chưa cải thiện rõ. Channel-wise modulation toàn cục vẫn không thay thế được early fusion — SNV là tín hiệu điểm tại center, cần biểu diễn theo vị trí trong tensor.

### `notebooks/13_kaggle_train_sparse_alt_8ch_cnn_601.ipynb`

Ablation trực tiếp để kiểm tra giả thuyết: phần thắng của 9-channel đến từ ALT toàn chuỗi hay chỉ từ ALT tại center. Bản Kaggle standalone, cùng ràng buộc path và cờ tăng tốc như notebook 11/12.

**Lý do thử nghiệm — phân tích fusion:**

| Thiết kế | ALT trong CNN | Cơ chế | Nhược điểm |
|---|---|---|---|
| Full 9-channel (nb10) | Toàn bộ 601 bp | Early fusion | ~99.8% ALT positions là redundant copy của REF |
| Compact 4-channel (nb11) | Không có | Late fusion (vector) | CNN không thấy mutation trong convolution |
| Sparse ALT 8-channel (nb13) | Chỉ tại center | Early fusion | BatchNorm channel 4-7 rất thưa ở epoch đầu |

**Thiết kế:**

- Input 8 channel: `REF one-hot 4ch (toàn bộ 601 bp) + ALT one-hot 4ch (zeros trừ tại center)`.
- Bỏ marker channel: ALT nonzero đã đánh dấu center.
- Encode on-the-fly từ `X_ref_index` (0.80 GB) + `alt_base` từ metadata. Không lưu dense tensor riêng.
- `MODEL_VARIANT`:
  - `"sparse_alt"`: `DilatedClinVarCNN` chuẩn giống nb10, không có center_vector fusion, không SE/gating/FiLM.
  - `"sparse_alt_multibranch"`: `MultiBranchDilatedClinVarCNN` — nhánh high-resolution bảo vệ tín hiệu center, nhánh downsample học context xa/global, concat hai nhánh trước classifier.
- RC augment/TTA complement cả 4 REF channels lẫn 4 ALT channels đúng cách.
- `BATCH_SIZE = 1024` trên Kaggle (nhỏ hơn vì 8ch nhẹ hơn 17ch).

**Section 22 — Calibration plot + Platt scaling:**

- Load predictions + split_indices, predict lại trên validation set.
- Fit Platt scaling (logistic regression 2 params A, B) trên validation, apply lên test.
- Vẽ calibration curve 10 bins trước/sau, gap per bin, probability distribution shift.
- Bảng so sánh: ROC-AUC, PR-AUC (không đổi sau Platt), Brier score, ECE (kỳ vọng giảm).

**Kết quả:** chưa chạy. Kỳ vọng PR-AUC xấp xỉ full 9-channel vì conv kernel tại center thấy input giống hệt; nếu đúng, 8-channel sparse thay thế được 9-channel dense để giảm storage ~7×.

**Lưu ý khi chạy:**

- `SEQ_CHANNELS = 8`; dataset trả về 2-tuple `(x, label)` không phải 3-tuple như compact.
- ALT sparse channels rất thưa: BatchNorm channel 4-7 có mean gần 0, variance nhỏ ở epoch đầu. Nếu training bất ổn, thử tăng BatchNorm momentum hoặc thêm warmup.


### `scripts/build_refseq_context_parquet.py` và `scripts/build_ref_sequence_parquet.py`

Các script thử nghiệm lưu context dạng `ref_seq` string Parquet thay vì dense one-hot `.npy`.

Mục tiêu:

- Giảm storage khi muốn lưu context dài 2001/4001/8001 bp.
- Đẩy dataset lên Kaggle nhẹ hơn.
- Encode one-hot on-the-fly trong `Dataset` khi train CNN.

Ước tính với khoảng 460k variants, `ref_seq` 8001 bp dạng Parquet `zstd` có thể ở mức khoảng `0.8-1.2 GB`, nhỏ hơn nhiều so với dense one-hot 9-channel 8001 bp khoảng `33 GB`.

### `notebooks/03_cnn_interpretability.ipynb`

Tính saliency gradient cho CNN 9-channel để xem model nhạy với vị trí/kênh nào quanh mutation.

Output:

- `processed/interpretability/cnn_saliency_center_vs_edges.parquet`
- `processed/interpretability/cnn_saliency_channel_importance.parquet`

Saliency chỉ là heuristic giải thích model, không phải bằng chứng nhân quả sinh học.

### `notebooks/07_build_clinvar_tabular_dataset.ipynb`

Tạo dataset tabular riêng cho XGBoost từ `Data/variant_summary.txt`.

Output hiện tại:

- `processed/clinvar_tabular_full.parquet`: shape `(1,426,185, 55)`.
- `processed/clinvar_tabular_unique_variant.parquet`: shape `(1,426,185, 55)`.
- `processed/clinvar_tabular_cnn_aligned.parquet`: shape `(460,488, 55)`, aligned với dataset CNN.

Notebook này giữ nhiều cột ClinVar gốc hơn pipeline CNN và thêm feature suy ra như review status đơn giản hóa, số submitter, năm đánh giá, độ dài allele, nhóm origin.

### `notebooks/04_xgboost_tabular_baseline.ipynb`

Train XGBoost baseline trên feature tabular nhẹ.

Notebook tránh đưa trực tiếp các cột label/leakage như `ClinicalSignificance`, `ClinSigSimple`, `Y` vào feature X. Có feature importance và SHAP sample nhỏ.

Output:

- `models/clinvar_xgboost_tabular_baseline.joblib`
- `processed/xgboost_tabular_baseline_test_predictions.parquet`
- `processed/xgboost_tabular_baseline_threshold_table.parquet`
- `processed/xgboost_tabular_baseline_feature_importance.parquet`
- `processed/xgboost_tabular_baseline_shap_importance.parquet`

### `notebooks/04_1_xgboost_optuna_tuning.ipynb`

Tune XGBoost bằng Optuna trên subset để tiết kiệm compute, sau đó train final model.

Artifact hiện tại cho thấy model Optuna trên `gene_group` split đang kém sequence CNN rõ rệt.

Output:

- `models/clinvar_xgboost_optuna_tuned.joblib`
- `processed/xgboost_optuna_tuned_test_predictions.parquet`
- `processed/xgboost_optuna_tuned_threshold_table.parquet`
- `processed/xgboost_optuna_tuned_trials.parquet`

### `notebooks/05_vep_consequence_annotation.ipynb`

Chuẩn bị VCF và workflow chạy Ensembl VEP local qua Docker để lấy `consequence`.

Output đã có:

- `processed/clinvar_for_vep.vcf`: `460,488` variants.

Lưu ý trạng thái hiện tại: file annotation tổng hợp đang có `consequence = unknown` cho toàn bộ dòng, nên VEP consequence thực tế chưa được merge vào artifact cuối.

### `notebooks/06_gnomad_cadd_annotation.ipynb`

Chuẩn bị query variant list và merge annotation ngoài:

- `gnomAD_AF`
- `gnomAD_AF_popmax`
- `CADD_RAW`
- `CADD_PHRED`
- `SpliceAI_max`

Output hiện tại:

- `Data/annotations/clinvar_annotation_query.tsv`
- `Data/annotations/clinvar_annotation_query.bed`
- `Data/annotations/clinvar_annotation_query.parquet`
- `Data/annotations/variant_annotations.parquet`
- `Data/annotations/variant_annotations.csv`

Lưu ý trạng thái hiện tại: `gnomAD_AF`, `CADD_PHRED`, `SpliceAI_max` đều đang null trong artifact đã lưu. Cần đặt file gnomAD/CADD đã lọc vào `Data/annotations/` rồi chạy lại notebook nếu muốn dùng annotation thật.

## Đánh giá notebook 10–13

### Tổng quan chuỗi thử nghiệm

Bốn notebook này tạo thành một chuỗi ablation có hệ thống, mỗi notebook trả lời đúng một câu hỏi về input representation:

```text
nb10 (9-ch full)  →  nb11 (4-ch compact late-fusion)  →  nb12 (4-ch + FiLM)  →  nb13 (8-ch sparse early-fusion)
     baseline              ablation: bỏ ALT khỏi CNN          ablation: global gate          ablation: ALT tại đúng vị trí
```

### Đánh giá từng notebook

#### Notebook 10 — Full 9-channel dilated CNN

**Điểm mạnh:**

- Pipeline hoàn chỉnh nhất: split logic nghiêm ngặt (genome_block + coordinate purge + sequence hash purge), hard-negative mining, RC aug/TTA, artifact versioning rõ ràng.
- Metric tốt nhất trong chuỗi: PR-AUC `0.9124`, F1 `0.8178`.
- Code modular, có thể dùng lại model/dataset/helper cho inference mà không cần retrain.
- Cell error analysis có thể nhận diện systematic failure (label noise single-submitter vs lỗi model thực).

**Điểm cần chú ý:**

- Dense tensor `X_ref_alt_marker_601_fixed_refseq.npy` (`7.19 GB`) không thể đẩy lên Kaggle free tier — phải dùng bản standalone nb09 hoặc encode on-the-fly.
- `CONFIG["scheduler"] = "none"` và `CONFIG["grad_clip_norm"] = 0.0` giữ đúng cấu hình của run tốt nhất cũ; chưa thử scheduler hay clipping trên split nghiêm ngặt mới.
- Chưa có calibration plot: probability output chưa được kiểm tra mức độ miscalibration.
- Kết quả test tốt nhưng chưa có multi-seed (chỉ seed 42); không rõ variance theo split.

#### Notebook 11 — Compact REF + center vector (local project)

**Điểm mạnh:**

- Pipeline self-contained từ raw: không phụ thuộc dense tensor cũ, tự filter + build compact cache.
- Tách biệt rõ `LABEL_MODE`, `MODEL_VARIANT`, `SPLIT_MODE` → dễ ablation.
- Có attribution heatmap để kiểm tra model học từ center hay học vùng xa (quan trọng cho giải thích sinh học).
- Cell so sánh 3 model (full / compact no-gate / compact SE-gate) trực tiếp trong notebook.

**Điểm cần chú ý:**

- SE-gate (`variant_gated_se`) chỉ cải thiện `0.8858 → 0.8870` PR-AUC so với compact no-gate — gating toàn cục chưa hiệu quả.
- Compact là **late fusion**: CNN không thấy ALT trong convolution → giải thích khoảng cách `0.025` PR-AUC với nb10.
- Có 2 notebook riêng (local `11_train_compact` và Kaggle `11_kaggle_train_compact`); logic trùng lặp nhiều — nếu sửa một cần nhớ sync sang bản kia.

#### Notebook 12 — FiLM/gating mạnh (Kaggle)

**Điểm mạnh:**

- Thử nghiệm FiLM đúng cách: last-layer zero-init để training bắt đầu giống compact baseline, không bị destabilized.
- Áp dụng FiLM tại **tất cả** 7 dilated blocks (`gate_placement="all"`) — đưa biến thể vào sớm và nhiều tầng.
- Giữ lại SE-gate cũ để so sánh ablation.

**Điểm cần chú ý:**

- Kết quả (`0.8874`) gần như bằng compact baseline (`0.8870`) và SE-gate (`0.8858`) — cải tiến không có ý nghĩa thống kê.
- Kết luận mạnh: **global channel modulation không thay thế được positional ALT information**. FiLM biết *loại* biến thể nhưng không biết *vị trí* → mâu thuẫn nội tại.
- Cần benchmark thêm với `FILM_HIDDEN_DIM` lớn hơn hoặc spatial FiLM (modulate theo vị trí, không chỉ theo channel).

#### Notebook 13 — Sparse ALT 8-channel (Kaggle, chưa chạy)

**Điểm mạnh:**

- Hypothesis đúng và thuyết phục: giữ ALT tại đúng vị trí center = early fusion mà không cần redundant copy ALT ở 600 vị trí còn lại.
- Nếu kỳ vọng đúng (PR-AUC ≈ nb10), giảm storage dense tensor từ `7.19 GB` xuống `0.80 GB` (−89%).
- `MultiBranchDilatedClinVarCNN` là thử nghiệm thú vị: nhánh high-res bảo vệ tín hiệu center, nhánh downsample học context xa.
- Section 22 calibration + Platt scaling là bổ sung quan trọng còn thiếu ở các notebook trước.

**Điểm cần chú ý:**

- **Chưa có kết quả** — đây là notebook duy nhất trong chuỗi chưa được chạy.
- BatchNorm trên channel 4-7 (ALT sparse) sẽ không ổn định ở epoch đầu: mean ≈ 0, variance rất nhỏ do hầu hết vị trí là 0. Nên theo dõi loss epoch 1 kỹ.
- Nếu PR-AUC thấp hơn nb10 rõ rệt, điều đó sẽ chứng minh phần thắng của 9-channel đến từ ALT redundant ở nhiều vị trí (context spreading) chứ không chỉ từ early fusion tại center.

### So sánh metric tổng hợp

| Notebook | Input | Fusion | Test PR-AUC | Test ROC-AUC | F1 @ val thr | Storage tensor |
|---|---|---|---:|---:|---:|---|
| nb10 | REF+ALT+marker 9ch + PE 8ch = 17ch | Early (toàn chuỗi) | **0.9124** | **0.9839** | **0.8178** | 7.19 GB |
| nb11 compact | REF 4ch + center vector 8dim | Late (vector) | 0.8870 | 0.9782 | 0.7946 | 0.80 GB |
| nb11 SE-gate | REF 4ch + SE gate | Late + global gate | 0.8858 | 0.9781 | 0.7940 | 0.80 GB |
| nb12 FiLM | REF 4ch + FiLM (tất cả blocks) | Late + global modulation | 0.8874 | 0.9780 | 0.7943 | 0.80 GB |
| nb13 sparse | REF 4ch + ALT sparse 4ch = 8ch | Early (chỉ center) | chưa chạy | chưa chạy | chưa chạy | 0.80 GB |

### Kết luận và hướng tiếp theo

1. **Gap early vs late fusion là `~0.025` PR-AUC** — có ý nghĩa thực tiễn (tăng ~800 true positives trong test set 216k dòng).
2. **Global modulation (SE/FiLM) không lấp được gap** — cần positional mutation signal trực tiếp trong tensor.
3. **Notebook 13 là thử nghiệm quyết định**: nếu sparse ALT 8-channel đạt PR-AUC ≥ 0.910, có thể bỏ dense 9-channel và tiết kiệm 89% storage.
4. **Bước sau khi có kết quả nb13**: chạy Platt calibration cho nb10 và nb11 để so sánh ECE; thử Siamese REF/ALT hoặc thêm biological annotation (VEP consequence, CADD, SpliceAI) để vượt trần sequence-only.

## Kết quả model hiện tại

Các số dưới đây được đọc lại từ metrics/prediction artifact trong `processed/`.

### Benchmark nghiêm ngặt trên `fixed_refseq`

Bảng này dùng split nghiêm ngặt hơn random/gene split cũ: `genome_block` theo block 1Mb, purge các variant val/test nằm gần train/val trong `5000 bp`, và purge exact REF sequence trùng. Các metric có threshold đều dùng threshold chọn từ validation, không dùng `0.5` làm tiêu chuẩn chính.

| Model | Input | Split | Test ROC-AUC | Test PR-AUC | F1 @ val threshold | Precision | Recall | Confusion matrix |
|---|---|---|---:|---:|---:|---:|---:|---|
| Full dilated CNN 601 bp | `REF+ALT+marker+PE`, 17-channel | genome_block + purge + seq purge | **0.9839** | **0.9124** | **0.8178** | **0.7920** | **0.8454** | `[[186995, 5335], [3713, 20308]]` |
| Compact CNN 601 bp | `REF 4-channel + center REF/ALT vector` | genome_block + purge + seq purge | 0.9782 | 0.8870 | 0.7946 | 0.7767 | 0.8133 | `[[186714, 5616], [4485, 19536]]` |
| Compact SE/gating 601 bp | compact + channel gate từ center REF/ALT | Kaggle, cùng split logic | 0.9781 | 0.8858 | 0.7940 | 0.7780 | 0.8107 | chưa lưu local |
| Compact FiLM blocks 601 bp | compact + FiLM residual blocks | Kaggle, cùng split logic | 0.9780 | 0.8874 | 0.7943 | 0.7762 | 0.8133 | chưa lưu local |
| Full dilated CNN 601 bp | `REF+ALT+marker+PE`, 17-channel | chromosome split | 0.9755 | 0.8877 | 0.7899 | 0.8244 | 0.7582 | `[[217058, 5531], [8280, 25965]]` |
| Sparse ALT 8-ch CNN 601 bp | `REF 4-ch + ALT 4-ch sparse at center`, 8-channel | Kaggle, cùng split logic | chưa chạy | chưa chạy | chưa chạy | chưa chạy | chưa chạy | chưa chạy |

Nhận xét chính:

- Full 9-channel vẫn tốt nhất vì là early fusion: convolution đầu đã thấy `REF`, `ALT` và marker đúng tại center.
- Compact input sạch và nhẹ hơn nhiều (`X_ref_index_601_fixed_refseq.npy` khoảng `0.80 GB` so với dense 9-channel khoảng `7.19 GB`) nhưng giảm khoảng `0.025` PR-AUC.
- SE/gating và FiLM đưa REF/ALT vào sớm hơn compact baseline, nhưng chưa kéo lại được khoảng cách. Gating hiện là điều kiện hóa channel toàn cục, trong khi tín hiệu SNV là cục bộ tại center.
- Sparse ALT 8-ch là ablation kiểm tra liệu giữ ALT tại đúng vị trí center (early fusion) nhưng bỏ ALT redundant ở 600 vị trí còn lại có đủ hay không.
- Chromosome split nghiêm hơn về chromosome holdout nhưng phân bố chromosome có thể khác và test set khác; dùng để kiểm tra độ bền, không so trực tiếp tuyệt đối với genome-block split.

### Benchmark lịch sử trước `fixed_refseq`

Bảng dưới đây là benchmark cũ trên artifact 460k và một số dòng dùng threshold `0.5`. Giữ lại để theo dõi lịch sử cải tiến, nhưng kết luận chính nên ưu tiên bảng `fixed_refseq` phía trên.

| Model | Split | Input chính | ROC-AUC | PR-AUC | Accuracy @0.5 | Precision pathogenic @0.5 | Recall pathogenic @0.5 | F1 pathogenic @0.5 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| CNN alt-only | random row | `alt_seq` 4-channel | 0.6526 | 0.2454 | 0.5705 | 0.1772 | 0.6389 | 0.2774 |
| CNN ref+alt+marker | random row | 9-channel | 0.8482 | 0.5224 | 0.7933 | 0.3532 | 0.7233 | 0.4746 |
| Transformer small | random row | 9-channel | 0.8018 | 0.4353 | 0.7441 | 0.2954 | 0.7097 | 0.4172 |
| CNN ref+alt+marker | gene_group | 9-channel | 0.8478 | 0.5224 | 0.7932 | 0.3435 | 0.7351 | 0.4682 |
| CNN gene_group balanced | gene_group | 9-channel | 0.8397 | 0.5114 | 0.8594 | 0.4464 | 0.5627 | 0.4979 |
| CNN + positional encoding | gene_group | 17-channel | **0.8806** | **0.5722** | **0.8835** | **0.5272** | 0.5811 | **0.5529** |
| CNN + positional encoding 1001 bp | gene_group | 17-channel, 1001 bp | 0.8584 | 0.5702 | 0.7834 | 0.3775 | **0.7691** | 0.5064 |
| Dilated CNN + positional encoding 601 bp | gene_group | 17-channel, 601 bp | 0.9327 | 0.7745 | 0.9081 | 0.6693 | 0.7191 | 0.6933 |
| Dilated CNN + positional encoding 1001 bp | gene_group | 17-channel, 1001 bp | **0.9317** | **0.7732** | **0.9039** | 0.6491 | 0.7287 | **0.6866** |
| Dilated CNN + positional encoding + RC augment/TTA 601 bp | gene_group | 17-channel, 601 bp | 0.9557 | 0.8431 | 0.9154 | 0.6693 | 0.8188 | 0.7366 |
| Dilated residual CNN + RC augment/TTA 601 bp | gene_group | 17-channel, 601 bp | 0.9493 | 0.8224 | 0.9215 | 0.7197 | 0.7479 | 0.7335 |
| Dilated CNN + relative PE + RC augment/TTA 601 bp | gene_group | 17-channel, 601 bp | 0.9533 | 0.8381 | 0.9159 | 0.6733 | 0.8121 | 0.7362 |
| Dilated CNN + window attention + RC augment/TTA 601 bp | gene_group | 17-channel, 601 bp | **0.9589** | **0.8507** | 0.9148 | 0.6605 | **0.8440** | **0.7410** |
| XGBoost tabular baseline | tabular split | ClinVar metadata | 0.7265 | 0.3932 | 0.8929 | 0.9326 | 0.1002 | 0.1809 |
| XGBoost Optuna tuned | gene_group | ClinVar metadata + optional annotations | 0.6095 | 0.1700 | 0.8736 | 0.1538 | 0.0045 | 0.0087 |

Threshold analysis đáng chú ý:

- CNN + positional encoding best F1: threshold `0.4685`, precision `0.5020`, recall `0.6182`, F1 `0.5541`.
- CNN + positional encoding best F2: threshold `0.2739`, precision `0.3711`, recall `0.7867`, F2 `0.6428`.
- CNN + positional encoding 1001 bp best F1: threshold `0.5857`, precision `0.4860`, recall `0.6152`, F1 `0.5430`.
- CNN + positional encoding 1001 bp best F2: threshold `0.4745`, precision `0.3480`, recall `0.8072`, F2 `0.6386`.
- Dilated CNN 1001 bp best validation-F1 threshold `0.5797`: test precision `0.6867`, recall `0.6903`, F1 `0.6885`.
- Dilated CNN 1001 bp best validation-F2 threshold `0.2635`: test precision `0.5316`, recall `0.8290`, F1 `0.6478`.
- Dilated CNN 601 bp + RC augment/TTA best validation-F1 threshold `0.6226`: test precision `0.7629`, recall `0.7346`, F1 `0.7485`.
- Dilated CNN 601 bp + RC augment/TTA best validation-F2 threshold `0.4373`: test precision `0.6228`, recall `0.8564`, F1 `0.7212`.
- Dilated CNN + window attention 601 bp + RC augment/TTA best validation-F1 threshold `0.6954`: test precision `0.7967`, recall `0.7162`, F1 `0.7543`.
- Dilated CNN + window attention 601 bp + RC augment/TTA best validation-F2 threshold `0.4449`: test precision `0.6180`, recall `0.8703`, F1 `0.7228`.
- XGBoost tabular baseline best F1: threshold `0.1450`, precision `0.4927`, recall `0.3152`, F1 `0.3844`.

## Thử nghiệm và kết luận gần đây

### 1. Sửa FASTA accession version làm dataset tăng mạnh

Ban đầu pipeline chỉ giữ khoảng 460k SNV vì mapping FASTA hẹp. Sau khi dùng đúng accession version GRCh38 như `NC_000001.11`, `NC_000002.12`, ... số dòng match FASTA tăng lên khoảng `1.426M`.

Kết luận:

- 460k không phải do filter label, mà do mapping FASTA/contig chưa bao phủ hết.
- Các benchmark cuối nên dùng artifact `fixed_refseq`.

### 2. Split nghiêm ngặt hơn

Đã chuyển trọng tâm từ random/gene split sang:

- `genome_block`: chia theo block genome 1Mb.
- `purge_distance_bp = 5000`: loại val/test quá gần train/val.
- `sequence_purge_mode = exact_ref`: loại exact REF context trùng giữa split.

Lý do:

- Random split bị genomic proximity leakage nặng.
- Gene-group split loại gene overlap nhưng vẫn có thể để các variant rất gần nhau hoặc overlapping gene lọt qua.
- Genome-block/purge làm bài toán gần với đánh giá tổng quát hóa vùng genome hơn.

### 3. Giảm 9-channel xuống compact 4-channel + fusion

Mục tiêu compact:

```text
REF sequence 601 bp -> CNN feature extractor
center REF/ALT vector -> classifier
```

Ưu điểm:

- Nhẹ hơn nhiều về storage.
- Tránh lặp ALT one-hot trên toàn bộ chuỗi.
- Dễ giải thích hơn vì mutation luôn ở center.

Kết quả:

- PR-AUC giảm `0.9124 -> 0.8870`.
- F1 giảm `0.8178 -> 0.7946`.
- False negative tăng `3713 -> 4485`.

Diễn giải:

- Full 9-channel là early fusion: kernel convolution thấy `REF`, `ALT`, marker center ngay từ layer đầu.
- Compact là late fusion: CNN trích feature từ REF trước, ALT chỉ vào classifier cuối.
- Vì SNV là biến đổi cục bộ tại center, việc đưa `ALT/marker` vào đúng vị trí rất có lợi.

### 4. SE/gating và FiLM chưa tăng rõ

Đã thử:

- SE/gating nhẹ: center REF/ALT sinh gate theo channel.
- FiLM blocks: center REF/ALT sinh `gamma/beta` cho nhiều residual dilated blocks.

Kết quả đều xấp xỉ compact baseline:

- SE/gating: test PR-AUC `0.8858`.
- FiLM blocks: test PR-AUC `0.8874`.
- Compact baseline: test PR-AUC `0.8870`.

Diễn giải:

- FiLM đưa thông tin biến thể vào sớm hơn, nhưng chủ yếu là channel-wise/global modulation.
- SNV cần local explicit mutation representation tại center.
- Nếu muốn giữ compact nhưng kéo metric lên, cần đưa mutation vào tensor theo vị trí, ví dụ ALT-at-center hoặc local mutation embedding, thay vì chỉ điều kiện hóa toàn cục.

### 5. Reverse-complement augment/TTA và dilated CNN là cải tiến quan trọng

Các thử nghiệm cũ cho thấy tăng mạnh nhất đến từ:

- Dilated CNN: tăng receptive field mà không pooling mất resolution quá sớm.
- RC augment/TTA: giúp model không phụ thuộc chiều đọc DNA.
- Hard-negative mining: tăng thêm PR-AUC/F1 ở giai đoạn fine-tune.

Residual/dilated residual không tự động tốt hơn. Window attention có lợi trong gene-group benchmark cũ, nhưng trên benchmark strict hiện tại chưa phải best chính.

## Ý tưởng tiếp theo

### Ưu tiên 1: local explicit mutation embedding → đã implement (notebook 13)

Thiết kế nằm giữa full 9-channel và compact:

```text
4 REF channels (toàn bộ 601bp)
+ 4 ALT-at-center channels, chỉ nonzero tại center (sparse)
= 8 channels, bỏ marker vì ALT nonzero đã đánh dấu center
```

Đã implement trong `notebooks/13_kaggle_train_sparse_alt_8ch_cnn_601.ipynb`. Chưa chạy, cần benchmark trên Kaggle hoặc local.

Mục tiêu: giữ thông tin mutation tại đúng vị trí, nhưng không lặp ALT trên toàn 601 bp. Đây là ablation trực tiếp để kiểm tra liệu phần thắng của 9-channel đến từ ALT toàn chuỗi hay chỉ từ ALT tại center.

### Ưu tiên 2: Siamese REF/ALT CNN

Thiết kế:

```text
REF sequence -> shared CNN -> feature_ref
ALT sequence -> shared CNN -> feature_alt
classifier(feature_ref, feature_alt, feature_alt - feature_ref, abs(diff), center REF/ALT)
```

Lý do:

- Gần với cách các model variant-effect như DeepSEA/Basenji/Enformer thường score variant: so sánh REF và ALT.
- Không cần giữ ALT 4 kênh lặp toàn chuỗi trong input.
- Cho phép giải thích bằng `feature_alt - feature_ref`.

Nhược điểm: chậm hơn gần 2 lần vì chạy shared CNN cho cả REF và ALT.

### Ưu tiên 3: CNN transfer learning

Pipeline đề xuất:

1. Pretrain CNN backbone trên FASTA GRCh38 bằng masked-base prediction hoặc corrupted-base recovery.
2. Load backbone vào model SNV.
3. Freeze backbone 1-2 epoch, train classifier.
4. Unfreeze toàn bộ, fine-tune với learning rate nhỏ cho backbone.

Mục tiêu: để CNN học DNA grammar/motif từ genome trước, sau đó mới học pathogenicity từ ClinVar.

### Ưu tiên 4: thêm annotation sinh học

Sequence-only có trần hiệu năng vì pathogenicity thường phụ thuộc vào gene, transcript, consequence, protein domain, allele frequency và bằng chứng clinical. Các nguồn nên cân nhắc:

- VEP consequence, codon/amino-acid change, transcript consequence.
- dbNSFP: SIFT, PolyPhen, MutationTaster, REVEL, AlphaMissense, conservation scores.
- CADD/SpliceAI/gnomAD AF.

Hướng này có khả năng tăng metric mạnh hơn việc tiếp tục phức tạp hóa CNN thuần sequence.

### Ưu tiên 5: giải thích model

Đã thêm các cell:

- Saliency/heatmap quanh center.
- Tỉ lệ saliency trong `±10`, `±50`, `±150 bp`.
- Error analysis case model tự tin sai.
- ReviewStatus cross-check để phân biệt nhiễu label single submitter với lỗi thực sự của model.

Nên đưa các bảng này vào báo cáo cuối kỳ vì chúng trả lời được câu hỏi model học motif gần center hay học vùng genome quen thuộc.

## Artifact chính

### Models

- `models/clinvar_altseq_cnn_demo_pytorch.pt`
- `models/clinvar_ref_alt_marker_cnn_demo_pytorch.pt`
- `models/clinvar_transformer_small_pytorch.pt`
- `models/clinvar_ref_alt_marker_cnn_gene_group_pytorch.pt`
- `models/clinvar_ref_alt_marker_cnn_gene_group_balanced_pytorch.pt`
- `models/clinvar_cnn_gene_group_positional_encoding_pytorch.pt`
- `models/clinvar_cnn_gene_group_positional_encoding_601_dilated_pytorch.pt`
- `models/clinvar_cnn_gene_group_positional_encoding_601_dilated_rcaug_rctta_pytorch.pt`
- `models/clinvar_cnn_gene_group_positional_encoding_601_dilated_residual_rcaug_rctta_pytorch.pt`
- `models/clinvar_cnn_gene_group_relative_positional_encoding_601_dilated_rcaug_rctta_pytorch.pt`
- `models/clinvar_cnn_gene_group_positional_encoding_601_dilated_window_attention_rcaug_rctta_pytorch.pt`
- `models/clinvar_cnn_gene_group_positional_encoding_601_fixed_refseq_dilated_rcaug_rctta_genome_block1000000_purged5000_seqpurge_exact_ref_pytorch.pt`
- `models/clinvar_cnn_gene_group_positional_encoding_601_fixed_refseq_dilated_rcaug_rctta_chromosome_split_pytorch.pt`
- `models/clinvar_compact_ref_alt_center_nopos_601_fixed_refseq_dilated_genome_block1000000_purged5000_seqpurge_exact_ref_pytorch.pt`
- `models/clinvar_cnn_gene_group_positional_encoding_1001_pytorch.pt`
- `models/clinvar_cnn_gene_group_positional_encoding_1001_dilated_pytorch.pt`
- `models/clinvar_xgboost_tabular_baseline.joblib`
- `models/clinvar_xgboost_optuna_tuned.joblib`

### Processed data

- `processed/clinvar_training_metadata.parquet`
- `processed/clinvar_context_201.parquet`
- `processed/clinvar_filtered_step8.parquet`
- `processed/clinvar_tabular_full.parquet`
- `processed/clinvar_tabular_unique_variant.parquet`
- `processed/clinvar_tabular_cnn_aligned.parquet`
- `processed/X_alt_201.npy`
- `processed/X_ref_alt_marker_201.npy`
- `processed/X_ref_alt_marker_601.npy`
- `processed/X_ref_alt_marker_601_fixed_refseq.npy`
- `processed/X_ref_index_601_fixed_refseq.npy`
- `processed/X_ref_alt_marker_1001.npy`
- `processed/y.npy`
- `processed/y_601.npy`
- `processed/y_601_fixed_refseq.npy`
- `processed/y_compact_ref_alt_center_601_fixed_refseq.npy`
- `processed/y_1001.npy`

### Predictions and reports

- `processed/cnn_demo_*`
- `processed/cnn_gene_group_*`
- `processed/cnn_gene_group_positional_encoding_601_*`
- `processed/cnn_gene_group_positional_encoding_601_fixed_refseq_*`
- `processed/compact_ref_alt_center_nopos_601_fixed_refseq_*`
- `processed/cnn_gene_group_positional_encoding_1001_*`
- `processed/transformer_small_*`
- `processed/xgboost_tabular_baseline_*`
- `processed/xgboost_optuna_tuned_*`
- `processed/interpretability/*`

## Cách chạy lại

Môi trường notebook hiện dùng Python có các package chính:

- `numpy`
- `pandas`
- `pyarrow`
- `scikit-learn`
- `torch`
- `xgboost`
- `optuna`
- `shap`
- `pyfaidx`
- `tqdm`
- `matplotlib`
- `seaborn`

Trên máy hiện tại có conda env `test-env` đọc được các artifact:

```bash
conda activate test-env
jupyter lab
```

Thứ tự chạy lại gợi ý:

1. `notebooks/01_read_clinvar_variant_summary.ipynb`
2. Sequence model:
   - `notebooks/02_cnn_demo.ipynb`
   - `notebooks/02_2_cnn_split_leakage_audit.ipynb`
   - `notebooks/02_3_cnn_gene_group_split.ipynb`
   - `notebooks/02_4_cnn_positional_encoding.ipynb`
   - `notebooks/03_cnn_interpretability.ipynb`
   - Hoặc chạy thử context 1001 bp bằng script:
     ```bash
     python scripts/build_cnn_dataset_1001.py
     python scripts/train_cnn_gene_group_positional_1001.py --batch-size 256
     python scripts/train_cnn_gene_group_positional_1001.py --architecture dilated --batch-size 256
     ```
   - Build context 2001 bp cho benchmark attention nhìn xa:
     ```bash
     python scripts/build_cnn_dataset_1001.py --flank 1000
     ```
   - Command chuẩn tiếp theo: 2001 bp + purged gene-group split + scheduler/clipping:
     ```bash
     python scripts/train_cnn_gene_group_positional_1001.py \
       --length 2001 \
       --architecture dilated_window_attention \
       --positional-encoding sinusoidal \
       --positional-dim 8 \
       --rc-augment \
       --rc-tta \
       --batch-size 128 \
       --split-mode purged_gene_group \
       --purge-distance-bp 5000 \
       --scheduler cosine \
       --warmup-epochs 1 \
       --grad-clip-norm 1.0
     ```
   - 4001 bp chỉ nên build sau khi 2001 chạy ổn:
     ```bash
     python scripts/build_cnn_dataset_1001.py --flank 2000
     ```
   - Chạy lại best model 601 bp cũ theo gene-group benchmark:
     ```bash
     python scripts/train_cnn_gene_group_positional_1001.py \
       --length 601 \
       --architecture dilated_window_attention \
       --positional-encoding sinusoidal \
       --positional-dim 8 \
       --rc-augment \
       --rc-tta \
       --batch-size 256
     ```
   - Chạy multi-seed để kiểm tra overfit/split sensitivity:
     ```bash
     python scripts/train_cnn_gene_group_positional_1001.py \
       --length 601 \
       --architecture dilated_window_attention \
       --positional-encoding sinusoidal \
       --positional-dim 8 \
       --rc-augment \
       --rc-tta \
       --batch-size 256 \
       --random-state 43
     ```
     Với `random_state != 42`, output sẽ có hậu tố `_seed43`, `_seed44`, ... để không ghi đè artifact seed 42.
   - Benchmark strict `fixed_refseq` hiện nên chạy qua notebook:
     - `notebooks/10_train_best_cnn_601_dilated_rcaug_rctta.ipynb`: full 9-channel model chính.
     - `notebooks/11_train_compact_ref_alt_center_cnn_601.ipynb`: compact REF + center REF/ALT.
     - `notebooks/12_kaggle_train_compact_ref_alt_center_film_cnn_601.ipynb`: FiLM/gating mạnh hơn trên Kaggle.
     - `notebooks/13_kaggle_train_sparse_alt_8ch_cnn_601.ipynb`: sparse ALT 8-channel ablation + calibration plot.
   - Khi chạy Kaggle, dùng bản standalone:
     - `notebooks/11_kaggle_train_compact_ref_alt_center_cnn_601.ipynb`
     - `notebooks/12_kaggle_train_compact_ref_alt_center_film_cnn_601.ipynb`
     - `notebooks/13_kaggle_train_sparse_alt_8ch_cnn_601.ipynb`
     - Input phải nằm trong `/kaggle/input`; output/cache/model nằm trong `/kaggle/working`.
3. Tabular/annotation:
   - `notebooks/07_build_clinvar_tabular_dataset.ipynb`
   - `notebooks/05_vep_consequence_annotation.ipynb` nếu cần VEP và Docker/cache đã sẵn sàng
   - `notebooks/06_gnomad_cadd_annotation.ipynb` nếu đã có file gnomAD/CADD đã lọc
   - `notebooks/04_xgboost_tabular_baseline.ipynb`
   - `notebooks/04_1_xgboost_optuna_tuning.ipynb`

## Caveat hiện tại

- Random row split cho kết quả dễ optimistic vì gene/genomic neighborhood overlap cao. `gene_group` tốt hơn random row split nhưng vẫn có genomic proximity leakage; benchmark chính hiện nên ưu tiên `genome_block + coordinate purge + sequence purge`.
- Context dài tốn storage/compute hơn đáng kể nếu lưu dense one-hot `.npy`. Dataset `fixed_refseq` 601 bp dense 9-channel đã khoảng `7.19 GB`; 2001/4001 bp trên 1.4M dòng sẽ rất nặng. Nên ưu tiên lưu REF string/base-index và encode on-the-fly nếu muốn context dài.
- Compact/FiLM chưa vượt full 9-channel. Điều này không có nghĩa ý tưởng sai, mà cho thấy mutation cần biểu diễn cục bộ tại center; global gating chưa thay thế được early fusion `REF+ALT+marker`.
- Các artifact benchmark đã chạy trước đây dùng cấu hình imbalance cũ có cả sampler và `pos_weight` dạng sqrt. Code hiện đã tách thành `IMBALANCE_STRATEGY`/`--imbalance-strategy` để tránh double class-weight trong các run mới; nên rerun benchmark chính trước khi chốt số cuối cùng.
- Window attention từng là best trên gene-group benchmark cũ, nhưng không nên xem là kết luận cuối trên split nghiêm ngặt mới nếu chưa chạy lại cùng `fixed_refseq/genome_block`.
- Do đã thử nhiều architecture trên cùng project, có nguy cơ overfit ở cấp experiment/benchmark. Nên báo cáo mean/std multi-seed và nếu có thể làm temporal/external validation.
- External annotation chưa thật sự đầy đủ: `variant_annotations.parquet` hiện có `consequence = unknown` và các cột gnomAD/CADD/SpliceAI đang null.
- Bài toán đang là binary classification đã loại `Uncertain significance`, `Conflicting interpretations`, `not provided` và các nhãn mơ hồ khác.
- Probability output của các model hiện tại chưa calibrate: model thường overconfident ở vùng probability trung bình (0.3-0.7). Notebook 13 có cell Platt scaling + calibration plot để kiểm tra và chỉnh; nên chạy cell này sau mỗi run để đánh giá mức độ miscalibration.
- Đây là project ML nghiên cứu/thử nghiệm, chưa phải pipeline clinical-grade.
