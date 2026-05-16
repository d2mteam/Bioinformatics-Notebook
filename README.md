# ClinVar pathogenicity prediction

## Dự án đang làm gì

Dự án này đang xây một pipeline machine learning cho bài toán phân loại biến thể ClinVar dạng SNV thành 2 nhóm:

- `1`: `Pathogenic` / `Likely pathogenic`
- `0`: `Benign` / `Likely benign`

Nguồn dữ liệu chính là `Data/variant_summary.txt` từ ClinVar snapshot đầu tháng 5/2026 và FASTA tham chiếu GRCh38 trong `Data/ncbi_dataset/`. Pipeline hiện có 2 hướng mô hình:

- Sequence models: dùng DNA context quanh vị trí đột biến, encode thành tensor cho CNN/Transformer. Mốc tốt nhất hiện tại là 601 bp với dilated CNN + 1D window attention.
- Tabular models: dùng metadata ClinVar và annotation ngoài tùy chọn cho XGBoost.

Artifact hiện tại cho thấy pipeline đã chạy xong preprocessing, đã train nhiều model baseline, đã audit leakage của random split, và model tốt nhất hiện tại là CNN `gene_group` 601 bp dạng dilated + window attention, có positional encoding, reverse-complement augmentation/TTA và hard-negative mining.

## Dữ liệu và label hiện tại

Pipeline sequence sau khi filter tạo ra dataset chính:

- `processed/clinvar_training_metadata.parquet`: `460,488` biến thể, `15` cột metadata.
- `processed/X_alt_201.npy`: shape `(460488, 201, 4)`, one-hot sequence đột biến.
- `processed/X_ref_alt_marker_201.npy`: shape `(460488, 201, 9)`, gồm `ref_seq` 4 kênh, `alt_seq` 4 kênh, mutation marker 1 kênh.
- `processed/y.npy`: shape `(460488,)`.

Thử nghiệm context dài 1001 bp tạo thêm:

- `processed/clinvar_training_metadata_1001.parquet`: `460,414` biến thể.
- `processed/X_ref_alt_marker_1001.npy`: shape `(460414, 1001, 9)`, khoảng `3.9 GB`.
- `processed/y_1001.npy`: shape `(460414,)`.

Thử nghiệm context 601 bp tạo thêm:

- `processed/clinvar_training_metadata_601.parquet`: `460,443` biến thể.
- `processed/X_ref_alt_marker_601.npy`: shape `(460443, 601, 9)`, khoảng `2.4 GB`.
- `processed/y_601.npy`: shape `(460443,)`.

Phân bố label trong dataset sequence:

| Label | Ý nghĩa | Số dòng |
|---|---|---:|
| `0` | Benign / Likely benign | 401,054 |
| `1` | Pathogenic / Likely pathogenic | 59,434 |

Tỉ lệ pathogenic khoảng `12.9%`, nên PR-AUC và precision/recall của class pathogenic quan trọng hơn accuracy.

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

- weighted sampler cho class imbalance
- `pos_weight`
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
- `--random-state`: đổi gene-group split; output seed khác 42 được gắn hậu tố `_seedXX` để tránh ghi đè artifact chính.

Kết quả hiện tại cho thấy chỉ tăng window lên 1001 bp bằng CNN baseline chưa cải thiện. Cải tiến mạnh nhất đến từ 601 bp + dilated CNN + RC augment/TTA, sau đó 1D window attention tiếp tục tăng PR-AUC/F1.

Output chính:

- `processed/clinvar_context_601.parquet`
- `processed/clinvar_training_metadata_601.parquet`
- `processed/X_ref_alt_marker_601.npy`
- `processed/y_601.npy`
- `processed/clinvar_context_1001.parquet`
- `processed/clinvar_training_metadata_1001.parquet`
- `processed/X_ref_alt_marker_1001.npy`
- `processed/y_1001.npy`
- `models/clinvar_cnn_gene_group_positional_encoding_601_dilated_rcaug_rctta_pytorch.pt`
- `models/clinvar_cnn_gene_group_positional_encoding_601_dilated_window_attention_rcaug_rctta_pytorch.pt`
- `models/clinvar_cnn_gene_group_positional_encoding_1001_pytorch.pt`
- `models/clinvar_cnn_gene_group_positional_encoding_1001_dilated_pytorch.pt`
- `processed/cnn_gene_group_positional_encoding_601_dilated_window_attention_rcaug_rctta_*`
- `processed/cnn_gene_group_positional_encoding_1001_*`

### `notebooks/08_train_best_cnn_601_window_attention_rcaug_rctta.ipynb`

Notebook mới đóng gói best model hiện tại và cách dùng lại artifact:

- Cấu hình best: `601 bp + dilated_window_attention + sinusoidal positional encoding + RC augment + RC TTA`.
- Mặc định `RUN_TRAIN = False` để không vô tình ghi đè artifact.
- Có cell gọi lại script train đúng tham số khi cần train lại.
- Có cell đọc metrics/history, load checkpoint và predict theo dataset index.

Notebook này không chỉnh sửa notebook 01.

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

## Kết quả model hiện tại

Các số dưới đây được đọc lại từ prediction artifact trong `processed/`.

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
- `processed/X_ref_alt_marker_1001.npy`
- `processed/y.npy`
- `processed/y_601.npy`
- `processed/y_1001.npy`

### Predictions and reports

- `processed/cnn_demo_*`
- `processed/cnn_gene_group_*`
- `processed/cnn_gene_group_positional_encoding_601_*`
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
   - Chạy lại best model hiện tại:
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
3. Tabular/annotation:
   - `notebooks/07_build_clinvar_tabular_dataset.ipynb`
   - `notebooks/05_vep_consequence_annotation.ipynb` nếu cần VEP và Docker/cache đã sẵn sàng
   - `notebooks/06_gnomad_cadd_annotation.ipynb` nếu đã có file gnomAD/CADD đã lọc
   - `notebooks/04_xgboost_tabular_baseline.ipynb`
   - `notebooks/04_1_xgboost_optuna_tuning.ipynb`

## Caveat hiện tại

- Nhiều notebook đang hard-code `PROJECT_DIR = Path("/mnt/MyData/Bioinformatics/Project")`. Nếu move project, cần sửa path hoặc refactor sang path tương đối.
- Random row split cho kết quả dễ optimistic vì gene/genomic neighborhood overlap cao. Nên ưu tiên báo cáo kết quả `gene_group`.
- Context dài tốn storage/compute hơn đáng kể nếu lưu dense one-hot `.npy`. Nên ưu tiên lưu `ref_seq` string Parquet nếu muốn 2001/4001/8001 bp và encode one-hot on-the-fly.
- Window attention hiện là best mới trên seed 42, nhưng runtime tăng khoảng `15.9 -> 24.5` phút. Cần chạy thêm seed 43/44 hoặc holdout độc lập trước khi xem là kết luận chắc.
- Do đã thử nhiều architecture trên cùng project, có nguy cơ overfit ở cấp experiment/benchmark. Nên báo cáo mean/std multi-seed và nếu có thể làm temporal/external validation.
- External annotation chưa thật sự đầy đủ: `variant_annotations.parquet` hiện có `consequence = unknown` và các cột gnomAD/CADD/SpliceAI đang null.
- Bài toán đang là binary classification đã loại `Uncertain significance`, `Conflicting interpretations`, `not provided` và các nhãn mơ hồ khác.
- Đây là project ML nghiên cứu/thử nghiệm, chưa phải pipeline clinical-grade.
