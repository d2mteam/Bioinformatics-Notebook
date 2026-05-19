# Báo cáo tổng kết project ClinVar SNV pathogenicity prediction

## 1. Mục tiêu và model cuối

Project xây dựng mô hình học sâu để phân loại biến thể ClinVar dạng SNV thành 2 lớp:

- `1`: `Pathogenic` / `Likely pathogenic`
- `0`: `Benign` / `Likely benign`

Model cuối được chọn cho hướng sequence CNN là notebook 13:

```text
notebooks/13_kaggle_train_sparse_alt_8ch_cnn_601.ipynb
```

Thiết kế input của model cuối:

```text
REF one-hot 4 channels trên toàn bộ window 601 bp
+ ALT one-hot 4 channels chỉ bật tại vị trí center
= 8 channels
```

Lý do chọn model này làm hướng cuối:

- Giữ được tín hiệu đột biến tại đúng center ngay từ lớp convolution đầu tiên.
- Bỏ phần ALT lặp dư thừa ở toàn bộ chuỗi. Với SNV, ngoài center thì `ALT sequence` gần như giống `REF sequence`, nên full 9-channel lưu rất nhiều thông tin thừa.
- Không cần marker channel riêng, vì ALT sparse nonzero tại center đã tự đánh dấu vị trí mutation.
- Nhẹ hơn full dense 9-channel, phù hợp Kaggle hơn.
- Vẫn giữ kiến trúc CNN mạnh nhất hiện tại: dilated 1D CNN + RC augment/TTA + strict split.

Lưu ý quan trọng: notebook 13 chưa có full metric cuối cùng ổn định. Theo yêu cầu hiện tại, báo cáo tạm lấy metric của notebook 10 làm metric proxy cho notebook 13, vì notebook 10 dùng cùng logic sinh học chính `601 bp + dilated CNN + RC augment/TTA + strict split`, chỉ khác input full 9-channel thay vì sparse ALT 8-channel.

## 2. Cải tiến dữ liệu và đánh giá nghiêm ngặt hơn

### 2.1. Lọc ClinVar rõ ràng hơn (bỏ)

Pipeline ban đầu chỉ dùng một phần dữ liệu SNV đã lọc. Về sau, việc lọc được chuẩn hóa thành các điều kiện rõ ràng:

- Chỉ giữ `Assembly == GRCh38`.
- Chỉ giữ `Type == single nucleotide variant`.
- Chỉ giữ REF/ALT là base đơn thuộc `A/C/G/T`.
- Chỉ giữ nhãn có thể map nhị phân:
  - benign side: `Benign`, `Likely benign`, các nhãn gộp benign/likely benign.
  - pathogenic side: `Pathogenic`, `Likely pathogenic`, các nhãn gộp pathogenic/likely pathogenic.
- Có tùy chọn `LABEL_MODE`:
  - `all`: giữ cả nhãn `Likely`, dùng để tương thích benchmark chính.
  - `definitive_only`: chỉ giữ `Benign` và `Pathogenic`, bỏ nhãn likely để giảm nhiễu label.
- Kiểm tra REF allele với FASTA GRCh38 trước khi giữ mẫu.
- Loại context chứa base ngoài `A/C/G/T`, hoặc variant quá sát đầu/cuối contig không đủ window.

Điểm quan trọng nhất là sửa mapping RefSeq accession version. Trước đây mapping FASTA hẹp làm dataset chỉ còn khoảng `460k` SNV. Sau khi dùng đúng accession như:

```text
NC_000001.11, NC_000002.12, ...
```

dataset `fixed_refseq` tăng lên khoảng `1.426M` SNV có REF match FASTA.

### 2.2. Từ random split sang split chống leakage

Audit notebook 02_2 cho thấy random split bị optimistic:

- Gene overlap giữa train/test rất cao.
- Nhiều test variants nằm rất gần train variants.
- Một phần context REF trùng hoặc gần trùng giữa split.

Các bước siết split:

1. `gene_group split`: không để cùng `GeneSymbol` xuất hiện ở train/val/test.
2. `purged_gene_group`: sau khi split theo gene, loại val/test variant quá gần train theo genomic distance.
3. `chromosome split`: kiểm tra khả năng tổng quát hóa theo chromosome holdout.
4. `genome_block split`: chia genome thành block 1Mb rồi split theo block.
5. `coordinate purge`: loại val/test nếu nằm gần train/val dưới `5000 bp`.
6. `sequence purge`: loại exact REF context trùng giữa split.

Benchmark chính hiện tại dùng:

```text
genome_block_size_bp = 1,000,000
purge_distance_bp = 5000
sequence_purge_mode = exact_ref
```

Audit split của benchmark chính:

| Split | Rows | Benign/Likely benign | Pathogenic/Likely pathogenic |
|---|---:|---:|---:|
| Train | 1,002,376 | 883,610 | 118,766 |
| Validation | 205,864 | 180,524 | 25,340 |
| Test | 216,351 | 192,330 | 24,021 |

Khoảng cách gần nhất sau purge:

| Tập | Min distance | Median distance |
|---|---:|---:|
| Val tới train | 5,001 bp | 398,213 bp |
| Test tới train/val | 5,023 bp | 388,904 bp |

Điều này làm metric đáng tin hơn nhiều so với random split, vì model khó học thuộc vùng genome quen thuộc hơn.

### 2.3. Sửa xử lý mất cân bằng lớp

Tỉ lệ pathogenic chỉ khoảng `11.8%`, nên nếu dùng accuracy sẽ dễ ảo. Project đã chuyển trọng tâm sang:

- PR-AUC.
- ROC-AUC.
- Precision/recall/F1 tại threshold chọn từ validation.
- Confusion matrix tại threshold validation.

bỏ (Ngoài ra đã phát hiện nguy cơ double class-weight: vừa dùng weighted sampler, vừa dùng `pos_weight` trong `BCEWithLogitsLoss`. Cấu hình mới chỉ dùng một cơ chế tại một thời điểm:

- `weighted_sampler`: inverse-frequency sampler + `BCEWithLogitsLoss()` thường.
- `pos_weight`: sampler thường + `BCEWithLogitsLoss(pos_weight=...)`.
- `legacy_sqrt_sampler_pos_weight`: chỉ để tái lập kết quả cũ.

Các metric cũ cần được hiểu là legacy nếu được train trước khi sửa class-weight.)

## 3. Lịch sử nâng cấp từ CNN 1D gốc

### 3.1. Baseline alt-only 201 bp

Phiên bản đầu tiên chỉ dùng `alt_seq` one-hot 4-channel quanh vị trí mutation.

Kết quả random split cũ:

| Model | ROC-AUC | PR-AUC | F1 @0.5 |
|---|---:|---:|---:|
| CNN alt-only | 0.6526 | 0.2454 | 0.2774 |

Điểm yếu: model chỉ thấy chuỗi sau đột biến, không biết base gốc là gì và mutation nằm chính xác ở đâu. Với SNV, thông tin `REF -> ALT` là trung tâm của bài toán, nên alt-only mất quá nhiều tín hiệu.

### 3.2. REF + ALT + marker 9-channel

Nâng cấp đầu tiên rất mạnh là thêm:

```text
REF one-hot 4 channels
ALT one-hot 4 channels
marker center 1 channel
```

Kết quả random split cũ:

| Model | ROC-AUC | PR-AUC | F1 @0.5 |
|---|---:|---:|---:|
| CNN alt-only | 0.6526 | 0.2454 | 0.2774 |
| CNN REF+ALT+marker | 0.8482 | 0.5224 | 0.4746 |

Tăng mạnh vì model biết:

- Base tham chiếu là gì.
- Base thay thế là gì.
- Vị trí mutation nằm ở đâu.
- Motif xung quanh mutation là gì.

Đây là cải tiến nền tảng quan trọng nhất ở giai đoạn đầu.

### 3.3. Gene-group split

Sau audit leakage, model được đánh giá bằng gene-group split thay vì random row split.

| Model | Split | ROC-AUC | PR-AUC | F1 @0.5 |
|---|---|---:|---:|---:|
| CNN REF+ALT+marker | random | 0.8482 | 0.5224 | 0.4746 |
| CNN REF+ALT+marker | gene_group | 0.8478 | 0.5224 | 0.4682 |

Metric gần như không giảm nhiều, nhưng đánh giá trở nên nghiêm túc hơn vì giảm khả năng cùng gene xuất hiện ở cả train và test.

### 3.4. Positional encoding

Notebook 02_4 thêm 8 kênh sinusoidal positional encoding:

```text
9 channels sequence + 8 channels position = 17 channels
```

| Model | ROC-AUC | PR-AUC | F1 @0.5 |
|---|---:|---:|---:|
| CNN gene_group 9-channel | 0.8478 | 0.5224 | 0.4682 |
| CNN + positional encoding | 0.8806 | 0.5722 | 0.5529 |

Tại sao tăng:

- CNN tự nhiên có tính dịch chuyển cục bộ, nhưng bài toán SNV luôn có mutation ở center.
- Positional encoding giúp model phân biệt tín hiệu gần center với tín hiệu xa center.
- Motif gần mutation thường quan trọng hơn vùng rìa, nên thông tin khoảng cách tới center giúp classifier tốt hơn.

Về sau, positional encoding không phải lúc nào cũng tăng tiếp khi input đã được thiết kế lại compact/sparse, nhưng ở giai đoạn CNN 201 bp nó giúp rõ.

### 3.5. Mở rộng context 601/1001 bp

Project thử tăng vùng nhìn từ 201 bp lên 601 bp và 1001 bp.

Kết luận chính:

- Chỉ tăng length mà vẫn dùng CNN nông không đủ.
- Window dài hơn chỉ có ích khi model có receptive field đủ lớn.
- 601 bp là điểm cân bằng tốt: đủ nhìn motif/region quanh mutation nhưng chưa quá nặng storage/compute.

### 3.6. Dilated 1D CNN

Dilated CNN là cải tiến kiến trúc lớn nhất sau input 9-channel. Thay vì pooling liên tục làm mất chi tiết vị trí, dilated convolution tăng receptive field bằng khoảng cách lấy mẫu trong kernel:

```text
dilation = 1, 2, 4, 8, 16, 32, 64
```

Kết quả benchmark cũ:

| Model | ROC-AUC | PR-AUC | F1 @0.5 |
|---|---:|---:|---:|
| CNN + PE 201 bp | 0.8806 | 0.5722 | 0.5529 |
| Dilated CNN + PE 601 bp | 0.9327 | 0.7745 | 0.6933 |

Tại sao tăng mạnh:

- Model nhìn được xa hơn mà không cần giảm resolution quá sớm.
- Mutation ở center vẫn giữ được định vị.
- Các layer dilation nhỏ học motif gần mutation.
- Các layer dilation lớn học pattern xa hơn, ví dụ vùng bảo tồn, motif lặp, ngữ cảnh GC hoặc cấu trúc sequence rộng hơn.
- Global max/mean pooling cuối gom đặc trưng sau khi CNN đã lan truyền thông tin đủ xa.

Đây là cải tiến kiến trúc có tác động metric rõ nhất.

### 3.7. Reverse-complement augment và RC TTA

DNA có tính đối xứng reverse-complement. Cùng một biến thể có thể biểu diễn trên mạch ngược:

```text
A <-> T
C <-> G
sequence đảo chiều
```

Project thêm:

- `RC augment`: trong train, random dùng sequence gốc hoặc reverse-complement.
- `RC TTA`: khi evaluate/test, predict cả bản gốc và reverse-complement rồi lấy trung bình.

Kết quả benchmark cũ:

| Model | ROC-AUC | PR-AUC | F1 @0.5 |
|---|---:|---:|---:|
| Dilated CNN + PE 601 bp | 0.9327 | 0.7745 | 0.6933 |
| Dilated CNN + PE + RC augment/TTA 601 bp | 0.9557 | 0.8431 | 0.7366 |

Tại sao tăng:

- Giảm overfit vào orientation của FASTA.
- Ép model học motif sinh học tương đương trên hai mạch DNA.
- TTA làm prediction ổn định hơn, nhất là ở các case model nhạy với chiều sequence.

### 3.8. Residual/skip connection (bỏ)

Đã thử dilated residual CNN.

| Model | ROC-AUC | PR-AUC | F1 @0.5 |
|---|---:|---:|---:|
| Dilated CNN + RC 601 bp | 0.9557 | 0.8431 | 0.7366 |
| Dilated residual CNN + RC 601 bp | 0.9493 | 0.8224 | 0.7335 |

Kết quả không tăng. Có thể vì model hiện tại chưa đủ sâu để residual là nút thắt chính, hoặc residual làm giữ quá nhiều tín hiệu thô không cần thiết. Skip connection hữu ích về mặt training stability, nhưng không phải cải tiến thắng metric trong project này.

### 3.9. Window attention 1D

Window attention được thử để cho model trộn thông tin cục bộ sau CNN.

| Model | ROC-AUC | PR-AUC | F1 @0.5 |
|---|---:|---:|---:|
| Dilated CNN + RC 601 bp | 0.9557 | 0.8431 | 0.7366 |
| Dilated CNN + window attention + RC 601 bp | 0.9589 | 0.8507 | 0.7410 |

Tại sao có tăng:

- CNN học motif bằng kernel cố định.
- Attention cho phép vị trí trong cùng window chọn vị trí liên quan mạnh hơn.
- Có ích khi signal phụ thuộc tương tác giữa các motif gần nhau.

Nhưng mức tăng nhỏ hơn nhiều so với dilated CNN và RC augment/TTA. Attention cũng đắt hơn, đặc biệt khi context tăng 2001/4001 bp.

### 3.10. Hard-negative fine-tuning

Sau train chính, project mine các negative mà model dễ nhầm thành pathogenic rồi fine-tune thêm.

Mục tiêu:

- Giảm false positive tự tin.
- Ép decision boundary tập trung vào các benign khó.
- Cải thiện precision ở threshold validation.

Hard-negative có ích cho vận hành threshold, nhưng không phải nguồn tăng metric lớn nhất so với input/dilation/RC.

### 3.11. Training stabilization

Pipeline sau thêm:

- AdamW.
- Cosine scheduler hoặc OneCycle.
- Warmup.
- Gradient clipping.
- Log learning rate và grad norm.

Các thay đổi này chủ yếu làm train ổn định hơn, đặc biệt với 2001 bp/window attention. Chúng không phải “trick sinh học”, nhưng giúp tránh kết luận sai do training bất ổn.

## 4. Các hướng kiến trúc compact/gating gần đây
giữ mục cuối 4.3
### 4.1. Compact REF + center REF/ALT

Notebook 11 giảm input còn:

```text
REF sequence 4-channel -> CNN
center REF/ALT vector 8 chiều -> classifier cuối
```

Kết quả strict split:

| Model | Test ROC-AUC | Test PR-AUC | F1 @ val threshold | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Full 9-channel notebook 10 | 0.9839 | 0.9124 | 0.8178 | 0.7920 | 0.8454 |
| Compact notebook 11 | 0.9782 | 0.8870 | 0.7946 | 0.7767 | 0.8133 |

Metric giảm khoảng `0.0254` PR-AUC.

Giải thích:

- Compact đưa REF/ALT center vào classifier cuối, tức late fusion.
- CNN backbone khi trích đặc trưng chỉ nhìn REF sequence, chưa biết mutation là gì.
- Vì vậy các convolution đầu không thể học trực tiếp motif kiểu “base C ở reference đổi thành T tại motif này”.
- Full 9-channel tốt hơn vì mutation signal đi vào từ layer đầu.

### 4.2. SE/gating và FiLM

Để đưa thông tin mutation vào sớm hơn compact baseline, notebook 12 thử:

- SE/gating: dùng center REF/ALT để gate channel.
- FiLM blocks: dùng center REF/ALT sinh `gamma, beta` để điều kiện hóa feature map trong residual dilated blocks.

Kết quả Kaggle:

| Model | Test ROC-AUC | Test PR-AUC | F1 @ val threshold | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Compact SE/gating | 0.9781 | 0.8858 | 0.7940 | 0.7780 | 0.8107 |
| Compact FiLM blocks | 0.9780 | 0.8874 | 0.7943 | 0.7762 | 0.8133 |

FiLM/SE không tăng rõ so với compact baseline.

Giải thích:

- FiLM/gating điều chỉnh channel toàn cục, nhưng SNV là tín hiệu rất cục bộ tại center.
- Model vẫn không có một “điểm ALT” cụ thể trong chuỗi để convolution đầu học local pattern.
- Full 9-channel/sparse ALT thắng về nguyên lý vì mutation xuất hiện như một đặc trưng không gian tại center, không chỉ là vector phụ.

### 4.3. Sparse ALT 8-channel: model cuối

Notebook 13 sửa đúng điểm yếu của compact:

```text
REF 4-channel toàn chuỗi
ALT 4-channel chỉ tại center
```

So với full 9-channel:

- Giữ early fusion tại center.
- Bỏ ALT redundant ngoài center.
- Bỏ marker channel vì ALT sparse đã chỉ center.
- Dùng cache nhẹ `X_ref_index` thay vì dense full 9-channel.

Kỳ vọng:

- Metric gần notebook 10 vì convolution tại center vẫn thấy REF và ALT cùng vị trí.
- Storage/train I/O nhẹ hơn rõ rệt.
- Dễ giải thích hơn compact/FiLM vì mutation được biểu diễn trực tiếp trong chuỗi.này

## 5. Metric cuối tạm thời cho notebook 13 (bỏ qua mục lớn này này)

Vì notebook 13 chưa có full metric cuối, báo cáo này tạm dùng metric notebook 10 làm proxy cho model cuối `Sparse ALT 8-channel CNN 601 bp`.

Điều kiện đánh giá:

```text
Dataset: fixed_refseq 601 bp
Split: genome_block 1Mb + coordinate purge 5000 bp + exact REF sequence purge
Training: dilated CNN + RC augment/TTA
Threshold: chọn từ validation, không dùng 0.5 làm chuẩn
```

Metric proxy:

| Metric | Giá trị |
|---|---:|
| Best validation PR-AUC | 0.9052 |
| Test PR-AUC | 0.9124 |
| Test ROC-AUC | 0.9839 |
| Test accuracy tại threshold validation-F1 | 0.9582 |
| Test precision pathogenic | 0.7920 |
| Test recall pathogenic | 0.8454 |
| Test F1 pathogenic | 0.8178 |

Confusion matrix tại threshold validation-F1:

```text
               Pred benign   Pred pathogenic
True benign        186,995             5,335
True pathogenic      3,713            20,308
```

Diễn giải:

- Model bắt được khoảng `84.5%` pathogenic SNV trong test.
- Precision pathogenic khoảng `79.2%`, nghĩa là trong các biến thể model gọi là pathogenic, khoảng 4/5 là đúng theo label ClinVar.
- False negative còn `3,713`, đây là nhóm cần error analysis thêm vì bỏ sót pathogenic thường quan trọng hơn trong bài toán y sinh.
- False positive `5,335`, có thể chứa cả label nhiễu hoặc ClinVar case chưa chắc chắn.

Không nên dùng accuracy làm kết luận chính vì class imbalance mạnh. PR-AUC và precision/recall của pathogenic mới là chỉ số quan trọng hơn.

## 6. Cải tiến nào tăng metric nhiều nhất?

Xếp theo tác động thực nghiệm và ý nghĩa kỹ thuật:

| Hạng | Cải tiến | Tác động |
|---:|---|---|
| 1 | Thêm `REF+ALT+marker` thay vì alt-only | Tăng PR-AUC từ `0.2454` lên `0.5224` trên benchmark đầu |
| 2 | Dilated CNN + context 601 bp | Tăng PR-AUC từ `0.5722` lên `0.7745` so với CNN+PE 201 bp |
| 3 | RC augment/TTA | Tăng PR-AUC từ `0.7745` lên `0.8431` |
| 4 | Positional encoding giai đoạn đầu | Tăng PR-AUC từ `0.5224` lên `0.5722` |
| 5 | Window attention | Tăng nhẹ PR-AUC từ `0.8431` lên `0.8507` |
| 6 | Strict filtering/split | Không nhất thiết làm metric tăng, nhưng làm metric đáng tin hơn |
| 7 | SE/gating/FiLM | Chưa tăng rõ, nhưng giúp hiểu rằng late/global fusion chưa đủ cho SNV |
| 8 | Sparse ALT 8-channel | Chưa có metric riêng, nhưng là thiết kế cuối hợp lý nhất về input và storage |

Kết luận kiến trúc:

- Với SNV, điều quan trọng nhất là biểu diễn được phép biến đổi `REF -> ALT` tại đúng center.
- Dilated CNN mạnh vì nó giữ resolution trong khi vẫn mở rộng vùng nhìn.
- RC augment/TTA mạnh vì DNA có đối xứng reverse-complement tự nhiên.
- Attention/FiLM/gating chỉ hữu ích khi không làm mất tín hiệu mutation cục bộ.
- Sparse ALT 8-channel là hướng tốt nhất về thiết kế vì giữ được early fusion nhưng bỏ phần redundant.

## 7. Bản đồ notebook thuộc pipeline CNN sau khi đọc lại

Phần này chỉ giữ các notebook phục vụ pipeline CNN: preprocess sequence, CNN training/evaluation, interpretability CNN và Kaggle CNN.

| Notebook | Vai trò | Thông tin cần đưa vào báo cáo |
|---|---|---|
| `00_read_clinvar_variant_summary_clean.ipynb` | Bản clean của pipeline đọc ClinVar | Trùng chức năng với notebook 01: đọc `variant_summary`, filter label/GRCh38/SNV/review status, match FASTA, tạo context và tensor. |
| `01_read_clinvar_variant_summary.ipynb` | Pipeline preprocess đầu tiên | Tạo dataset sequence 201 bp ban đầu: `X_alt_201`, `X_ref_alt_marker_201`, `y`, metadata. Đây là nền cho CNN demo. |
| `02_cnn_demo.ipynb` | CNN 1D baseline | Train CNN 1D 9-channel trên random split. Cho thấy `REF+ALT+marker` tốt hơn alt-only. |
| `02_1_model_experiments.ipynb` | Model zoo nhỏ | So sánh `cnn_baseline`, `cnn_residual`, `transformer_small`. Transformer nhỏ không vượt CNN; do dữ liệu 201 bp và architecture nhỏ chưa tận dụng tốt sequence. |
| `02_2_cnn_split_leakage_audit.ipynb` | Audit leakage | Chứng minh random split/gene overlap/proximity leakage là vấn đề thật. Đây là lý do chuyển sang gene-group rồi genome-block purge. |
| `02_3_cnn_gene_group_split.ipynb` | Gene-group CNN | Không để cùng gene rơi vào nhiều split. Có hard-negative fine-tuning và threshold table. |
| `02_4_cnn_positional_encoding.ipynb` | Positional encoding | Thêm 8 kênh sinusoidal position, tăng PR-AUC ở giai đoạn CNN 201 bp. |
| `03_cnn_interpretability.ipynb` | Giải thích model | Tính saliency gradient, heatmap theo channel/vị trí, center-vs-edge saliency, ví dụ variant cụ thể. Dùng để kiểm tra model tập trung quanh mutation hay học vùng xa. |
| `08_train_best_cnn_601_window_attention_rcaug_rctta.ipynb` | Đóng gói best model cũ | Notebook wrapper cho `601 bp + dilated_window_attention + RC augment/TTA` trên benchmark cũ. Không phải final strict benchmark. |
| `09_kaggle_train_window_attention_from_raw.ipynb` | Kaggle standalone window-attention | Tự tìm raw `variant_summary` và FASTA, filter, build tensor dense, train `dilated_window_attention`. Dùng để chuyển pipeline lên Kaggle. |
| `10_train_best_cnn_601_dilated_rcaug_rctta.ipynb` | Benchmark strict full 9-channel | Model có metric tốt nhất đã chạy local: full 9-channel + PE + dilated CNN + RC augment/TTA + genome-block purge. Metric này đang làm proxy cho notebook 13. |
| `11_train_compact_ref_alt_center_cnn_601.ipynb` | Compact model project | Dùng REF sequence + center REF/ALT vector. Giảm storage mạnh nhưng giảm PR-AUC vì mutation được đưa vào muộn. Có error analysis/heatmap. |
| `11_kaggle_train_compact_ref_alt_center_cnn_601.ipynb` | Compact model Kaggle | Bản Kaggle của notebook 11, đọc input từ `/kaggle/input`, ghi cache/model vào `/kaggle/working`, tối ưu tốc độ bằng AMP và DataLoader workers. |
| `12_kaggle_train_compact_ref_alt_center_film_cnn_601.ipynb` | FiLM/gating Kaggle | Thử đưa center REF/ALT vào nhiều block CNN bằng FiLM. Không cải thiện rõ so với compact baseline. |
| `13_kaggle_train_sparse_alt_8ch_cnn_601.ipynb` | Model cuối | Sparse ALT 8-channel: REF toàn chuỗi + ALT chỉ tại center. Đây là thiết kế cuối vì giữ early fusion nhưng bỏ redundant ALT/marker. |

### 7.1. Nhánh preprocess và dữ liệu

Notebook `00_read_clinvar_variant_summary_clean.ipynb` và `01_read_clinvar_variant_summary.ipynb` là nền dữ liệu cho CNN. Chúng cho thấy ClinVar raw có rất nhiều loại label và variant type, nhưng project cố ý giữ phạm vi SNV nhị phân để bài toán rõ:

- Raw `Type` có cả deletion, duplication, indel, insertion, microsatellite, copy number gain/loss.
- Project chỉ giữ `single nucleotide variant`.
- Raw `ClinicalSignificance` có nhiều nhóm như `Uncertain significance`, `Conflicting classifications`, `drug response`, `risk factor`.
- Project chỉ giữ benign/pathogenic và likely tương ứng.

Điều này quan trọng khi bảo vệ project: model hiện tại không phải model tổng quát cho mọi biến thể ClinVar, mà là model SNV pathogenicity binary classifier.

### 7.2. Nhánh CNN sequence

Flow CNN phát triển theo thứ tự:

```text
01 preprocess
-> 02 CNN random split
-> 02_2 audit leakage
-> 02_3 gene_group split
-> 02_4 positional encoding
-> scripts build 601/1001/2001
-> 08 window attention wrapper
-> 10 strict fixed_refseq benchmark
-> 11 compact
-> 12 FiLM/gating
-> 13 sparse ALT final
```

Điểm đáng nói:

- Notebook 02 xác nhận CNN 1D học được tín hiệu phân loại từ sequence quanh SNV.
- Notebook 02_2 làm thay đổi cách đánh giá, vì random split có leakage vùng genome.
- Notebook 10 mới là mốc strict benchmark đáng tin nhất đã có metric local.
- Notebook 13 là thiết kế cuối hợp lý hơn về input, nhưng cần train full để thay metric proxy.

### 7.3. Nhánh interpretability

Notebook 03 và các cell cuối notebook 10/11/13 cung cấp phần giải thích:

- Saliency theo vị trí quanh center.
- Channel importance.
- Heatmap CNN-style.
- High-confidence wrong cases.
- Đối chiếu `GeneSymbol`, `REF/ALT`, `ClinicalSignificance`, `ReviewStatus`.

Ý nghĩa trong báo cáo cuối kỳ:

- Không chỉ báo metric, mà kiểm tra model có đang nhìn quanh mutation center hay không.
- Error analysis giúp phân biệt lỗi do label nhiễu/low review status với lỗi thật của model.
- Các case model tự tin sai là nguồn tốt để phân tích giới hạn của CNN hiện tại.

### 7.4. Nhánh Kaggle và khả năng tái lập

Notebook 09/11 Kaggle/12/13 chuyển pipeline từ local sang Kaggle:

- Input chỉ đọc từ `/kaggle/input`.
- Output/cache/model ghi vào `/kaggle/working`.
- Tự discover `variant_summary.txt` và FASTA GRCh38.
- Dùng AMP, batch lớn, DataLoader workers, `TRAIN_METRICS_MODE="loss_only"` để giảm thời gian train.
- Có smoke test trước full train.

Notebook 13 hiện có cache:

```text
filtered parquet: 1,426,477 rows
metadata: 1,426,427 rows
X_ref_index: (1,426,427, 601), uint8
y: (1,426,427,)
active labels: 1,257,967 benign vs 168,460 pathogenic
```

Split audit trong notebook 13 khớp benchmark strict:

```text
train: 1,002,376
val:     205,864
test:    216,351
genome block overlap: 0
nearest distance min: val 5001 bp, test 5023 bp
```

Điều này xác nhận notebook 13 đã dựng đúng dữ liệu và split, còn thiếu bước train full để có metric riêng.

## 8. Điểm còn cần làm trước khi chốt báo cáo cuối kỳ

1. Train full notebook 13 để thay metric proxy bằng metric thật.
2. Rerun notebook 10/11/13 cùng cấu hình `IMBALANCE_STRATEGY = "weighted_sampler"` để so sánh công bằng sau khi fix double class-weight.
3. Báo cáo thêm calibration:
   - Brier score.
   - ECE.
   - Platt scaling validation -> test.
4. Dùng error analysis cuối:
   - Case model tự tin sai.
   - Gene nào hay sai.
   - REF/ALT nào hay sai.
   - `ReviewStatus` của các case sai.
5. Nếu có thời gian, thêm saliency/heatmap trực tiếp cho notebook 13 sau khi có checkpoint final.

## 9. Kết luận

Project đã đi từ CNN 1D baseline đơn giản đến một pipeline sequence classification khá chặt:

- Dữ liệu được lọc và đối chiếu FASTA nghiêm ngặt hơn.
- Split được siết để giảm genomic leakage.
- Metric chuyển từ accuracy/F1@0.5 sang PR-AUC và threshold chọn từ validation.
- Kiến trúc được cải tiến qua nhiều bước: `REF+ALT+marker`, positional encoding, dilated CNN, RC augment/TTA, window attention, compact input, SE/gating, FiLM, và cuối cùng là sparse ALT 8-channel.

Model cuối nên trình bày là:

```text
Sparse ALT 8-channel Dilated CNN 601 bp
```

Metric tạm báo cáo:

```text
Test PR-AUC: 0.9124
Test ROC-AUC: 0.9839
Test F1 pathogenic tại threshold validation-F1: 0.8178
Precision: 0.7920
Recall: 0.8454
```

Đây là metric proxy từ notebook 10 cho tới khi notebook 13 được train full. Khi notebook 13 chạy xong, cần cập nhật lại bảng metric để tránh nhầm giữa kỳ vọng thiết kế và kết quả thực nghiệm.
