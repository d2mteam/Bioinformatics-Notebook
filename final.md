ĐẠI HỌC QUỐC GIA HÀ NỘI

TRƯỜNG ĐẠI HỌC CÔNG NGHỆ



**MÔN HỌC: TIN SINH HỌC ỨNG DỤNGDỰ ĐOÁN TÍNH GÂY HẠI CỦA BIẾN ĐỔI GEN THÔNG QUA DEEP LEARNINGLớp học phần:** INT3423 1

**Giảng viên:** Lê Sỹ Vinh

**Nhóm:** Nhóm 6

**Thành viên:** Phùng Tiến Dũng - 23020030

Dương Đình Minh - 23020110

HÀ NỘI - 2026

**MỤC LỤC**

[**1\. Đặt vấn đề 2**](#_qdwwra4jcqni)

[1.1. Bài toán dự đoán tính gây bệnh của đột biến gen 2](#_m1n3r0s3bam5)

[1.2. Tổng quan về các phương pháp chẩn đoán tự động hiện nay 3](#_xd19qyyt5hq8)

[1.3. Phát triển hệ thống PathoGen ứng dụng mạng nơ-ron sâu 3](#_i5x7zljsl9gb)

[**2\. Kiến thức cơ sở 4**](#_wtdcek7ixep1)

[2.1. Biến dị di truyền và tính gây bệnh 5](#_6xmfja6ki43)

[2.1.1. Đột biến đơn nucleotide (Single Nucleotide Variant - SNV) 5](#_dmet9gargin3)

[2.1.2. Tác động của đột biến lên chức năng Protein 6](#_2s7g6svz4mkh)

[2.2. Các cơ sở dữ liệu tin sinh học 7](#_mc38lmo4r5md)

[2.2.1. Cơ sở dữ liệu ClinVar (NCBI) 7](#_i7jui81tqbwe)

[2.2.2. Hệ gen tham chiếu GRCh38 8](#_ikq5n2fg4jai)

[2.3. Mạng thần kinh tích chập một chiều (1D-CNN) 9](#_k9n5mst0nclv)

[2.3.1. Giới thiệu chung về 1D-CNN trong phân tích chuỗi 9](#_wfuln8cmfffo)

[2.3.2. Các thành phần cốt lõi của 1D-CNN 10](#_qdiq3dv4ydv)

[**3\. Phương pháp dự đoán tính gây hại của đột biến gen thông qua CNN 12**](#_5qm693df0lgs)

[3.1. Dữ liệu 13](#_snkkasifndqx)

[3.1.1. Nguồn dữ liệu 13](#_c5swgxfag9eq)

[3.1.2. Lọc dữ liệu 13](#_y8vwduopdytg)

[3.1.3. Phương pháp mã hóa chuỗi DNA (Biểu diễn ma trận Tensor Sparse ALT 8-Channel) 14](#_wj5um767hu7r)

[3.1.4. Chiến lược chống rò rỉ dữ liệu 15](#_xi9yar7g5ylk)

[3.2. Kiến trúc mô hình mạng tích chập Dilated 1D-CNN và Cơ chế đối ngẫu 17](#_zc2mgf7eoy0d)

[3.2.1. Kiến trúc các lớp tích chập một chiều giãn nở 17](#_s5zkpwmdzdvv)

[3.2.2. Chiến lược học đối ngẫu trên mạch đôi DNA (RC Augment & RC TTA) 18](#_o33sc9v21rjt)

[3.3. Huấn luyện, Chiến lược xử lý mất cân bằng lớp và Kết quả thực nghiệm. 19](#_zf49jxdw3tfb)

[3.3.1. Chiến lược xử lý mất cân bằng phân phối lớp và Ổn định huấn luyện 19](#_a9868abssd4f)

[3.3.2. Lựa chọn hệ đo lường và Ngưỡng quyết định 20](#_abznfadq2lf9)

[3.3.3. Kết quả thực nghiệm và Ma trận nhầm lẫn 21](#_4de7llucqa5j)

[3.3.4. Khả năng diễn giải mô hình 22](#_3v1tn6hn0c0v)

[**4\. Kết luận 22**](#_7hc90l983j4n)

[**5\. Tài liệu tham khảo 24**](#_azk72pq4y3ko)

# Đặt vấn đề

Chương này trình bày bối cảnh nghiên cứu và tính cần thiết của bài toán dự đoán tính gây bệnh của các đột biến gen (Variant Pathogenicity) trong kỷ nguyên y học chính xác. Các nghiên cứu hiện nay trong y học chính xác đang tập trung giải quyết khối lượng dữ liệu khổng lồ từ việc giải trình tự gen, đặc biệt là việc phân loại các biến dị di truyền chưa rõ ý nghĩa (VUS). Bài báo cáo này trình bày nghiên cứu về việc ứng dụng mạng thần kinh tích chập một chiều (1D-CNN) nhằm tự động hóa quá trình phân loại đột biến, đồng thời cung cấp một nền tảng tra cứu và trực quan hóa dữ liệu sinh học trực quan cho người dùng cuối.

## Bài toán dự đoán tính gây bệnh của đột biến gen

Sự bùng nổ của công nghệ Giải trình tự gen thế hệ mới (Next-Generation Sequencing - NGS) đã tạo ra một cuộc cách mạng trong y sinh học, cho phép các bác sĩ và nhà nghiên cứu giải mã toàn bộ bản đồ di truyền của một cá nhân với chi phí ngày càng tối ưu. Tuy nhiên, thành tựu này lại kéo theo một thách thức to lớn về bài toán xử lý dữ liệu lớn (Big Data). Trung bình, hệ gen của một người bình thường chứa khoảng 4 đến 5 triệu biến dị di truyền so với hệ gen tham chiếu chuẩn.

Trong số hàng triệu biến dị đó, chỉ có một phần nhỏ đã được chứng minh lâm sàng là nguyên nhân trực tiếp làm phá vỡ chức năng cấu trúc của protein, dẫn đến các bệnh lý di truyền hiểm nghèo hoặc ung thư (nhóm Pathogenic). Một phần khác đã được xác nhận là các biến dị hoàn toàn vô hại (nhóm Benign). Vấn đề cốt lõi của y học hiện đại nằm ở phần còn lại: phần lớn các đột biến mới được phát hiện từ dữ liệu NGS lại rơi vào "vùng xám" và được phân loại là Nhóm biến dị chưa rõ ý nghĩa lâm sàng (Variants of Uncertain Significance - VUS).

Sự tồn tại của các VUS tạo ra những rào cản nghiêm trọng trong chẩn đoán lâm sàng. Khi bác sĩ không thể kết luận một đột biến là có hại hay vô hại, bệnh nhân sẽ mất đi cơ hội được áp dụng các liệu pháp điều trị phù hợp. Việc phân loại thủ công các VUS này đòi hỏi các thử nghiệm chức năng sinh hóa (in vitro/in vivo) cực kỳ đắt đỏ, tiêu tốn nhiều tháng làm việc trong phòng thí nghiệm. Do đó, nhu cầu cấp thiết hiện nay là phải phát triển các phương pháp tính toán mạnh mẽ, tận dụng trực tiếp các đặc trưng từ chuỗi trình tự DNA để dự đoán nhanh chóng và chính xác tính gây bệnh của các VUS ngay khi chúng được giải trình tự.

## Tổng quan về các phương pháp chẩn đoán tự động hiện nay

Để giải quyết "nút thắt cổ chai" mang tên VUS, nhiều bộ công cụ Tin sinh học phân tích dự đoán đã được phát triển trong thập kỷ qua. Các công cụ truyền thống tiêu biểu như SIFT, PolyPhen-2, hay CADD chủ yếu vận hành dựa trên các quy tắc bảo thủ tiến hóa (evolutionary conservation) hoặc các chỉ số lý hóa tĩnh của axit amin. Mặc dù đóng vai trò nền tảng, các phương pháp này đang bộc lộ nhiều điểm yếu: chúng phụ thuộc quá nhiều vào việc trích xuất đặc trưng thủ công, dễ bị thiên lệch đối với các gen đã được nghiên cứu kỹ, và thường đạt ngưỡng bão hòa về độ chính xác do không nắm bắt được những tương tác mang tính ngữ cảnh không gian phức tạp trên chuỗi DNA.

Gần đây, sự trỗi dậy của Trí tuệ nhân tạo đặc biệt là Deep Learning, đã tạo ra những bước đột phá chưa từng có trong sinh học tính toán. Với khả năng trích xuất các mẫu ẩn từ dữ liệu phi cấu trúc, các mô hình học sâu hiện đại có thể học toàn bộ chuỗi DNA ngữ cảnh xung quanh vị trí đột biến để đưa ra dự đoán.

Tuy nhiên, công nghệ hiện tại lại bộc lộ một điểm mù lớn về mặt kỹ nghệ phần mềm và ứng dụng thực tiễn. Hầu hết các mô hình AI y sinh tiên tiến đều tồn tại dưới dạng các tệp mã nguồn mở, hoạt động trên giao diện dòng lệnh (CLI) và đòi hỏi việc thiết lập môi trường máy chủ phức tạp. Điều này tạo ra một rào cản kỹ thuật khổng lồ đối với các bác sĩ lâm sàng và nhà nghiên cứu sinh học - những chuyên gia có nhu cầu sử dụng kết quả dự đoán hàng ngày nhưng lại không chuyên sâu về khoa học máy tính và lập trình hệ thống.

## Phát triển hệ thống PathoGen ứng dụng mạng nơ-ron sâu

Nhận thức rõ những hạn chế cả về mặt giới hạn của thuật toán truyền thống lẫn rào cản triển khai thực tế, nghiên cứu này đề xuất phát triển Hệ thống PathoGen - một giải pháp công nghệ toàn diện kết hợp chặt chẽ giữa lõi Trí tuệ nhân tạo và kiến trúc Kỹ nghệ phần mềm.

- Về mặt phân tích dữ liệu và thuật toán lõi: Nghiên cứu thiết kế và áp dụng kiến trúc Mạng thần kinh tích chập một chiều (1D-CNN) được huấn luyện trên cơ sở dữ liệu lâm sàng chuẩn ClinVar (của tổ chức NCBI). Khác biệt với các phương pháp cũ, mô hình này lấy đầu vào trực tiếp là các đoạn DNA tham chiếu (GRCh38) bao quanh vị trí đột biến. Qua các lớp tích chập, hệ thống tự động học hỏi và trích xuất các _motif_ sinh học cốt lõi để tính toán ra xác suất gây bệnh một cách chính xác nhất.
- Về mặt kiến trúc hệ thống ứng dụng: Mô hình AI được thiết kế dưới dạng Microservice và tích hợp vào một hệ thống phần mềm hoàn chỉnh. Lõi hệ thống được vận hành bởi một backend mạnh mẽ, chịu trách nhiệm xử lý logic và truy xuất cơ sở dữ liệu tốc độ cao. Kết hợp với đó là một giao diện người dùng trực quan, cho phép các bác sĩ và nhà nghiên cứu dễ dàng nhập thông tin biến dị, quản lý lịch sử phân tích trên quy mô lớn và nhận báo cáo cảnh báo lâm sàng theo thời gian thực.

Cách tiếp cận song song này không chỉ nâng cao độ tin cậy trong việc tự động hóa quy trình phân loại VUS, mà còn trực tiếp thu hẹp khoảng cách giữa các thuật toán Tin sinh học hàn lâm và nhu cầu ứng dụng thực tiễn tại các cơ sở khám chữa bệnh.

# Kiến thức cơ sở

Trong chương này, các khái niệm nền tảng phục vụ cho việc xây dựng hệ thống dự đoán tính gây bệnh của biến dị di truyền sẽ được trình bày chi tiết. Nội dung bao gồm cơ sở sinh học phân tử về đột biến gen, các kho dữ liệu tin sinh học cốt lõi được sử dụng làm nguồn huấn luyện, và kiến trúc của mạng thần kinh tích chập một chiều (1D-CNN) được ứng dụng trong xử lý dữ liệu chuỗi tuần tự. Đồng thời, nghiên cứu cũng đi sâu phân tích hai thành phần hạ tầng dữ liệu quan trọng nhất của bài toán: cơ sở dữ liệu lâm sàng ClinVar và hệ gen tham chiếu GRCh38.

## Biến dị di truyền và tính gây bệnh

### Đột biến đơn nucleotide (Single Nucleotide Variant - SNV)

Phân tử DNA (Deoxyribonucleic Acid) là vật chất mang thông tin di truyền của hầu hết các sinh vật, được cấu trúc dưới dạng chuỗi xoắn kép. Ngôn ngữ di truyền của DNA được viết bởi bốn loại chữ cái hóa học bao gồm: Adenine (A), Thymine (T), Guanine (G), và Cytosine (C). Trong bộ gen người, các nucleotide này liên kết với nhau tạo thành khoảng 3.2 tỷ cặp base, đóng vai trò như một bản thiết kế hoàn chỉnh quy định mọi hoạt động sống của tế bào.

Trong quá trình sinh trưởng, phân tử DNA phải liên tục trải qua quá trình tự sao chép để truyền đạt thông tin cho thế hệ tế bào tiếp theo. Mặc dù hệ thống enzyme sao chép (DNA polymerase) có cơ chế tự kiểm tra và sửa lỗi với độ chính xác cực cao, nhưng dưới tác động của các tác nhân môi trường (tia bức xạ, hóa chất) hoặc những sai sót ngẫu nhiên trong nội bào, các lỗi sao chép vẫn có thể lọt qua. Sự sai lệch trong trình tự nucleotide này được gọi chung là đột biến gen.

Trong đó, Đột biến đơn nucleotide (Single Nucleotide Variant - SNV) là dạng biến dị di truyền vi mô và phổ biến nhất. Hiện tượng này xảy ra khi một nucleotide đơn lẻ tại một tọa độ xác định trên chuỗi DNA bị thay thế bằng một nucleotide khác (ví dụ: gốc Cytosine bị thay thế bởi Thymine). Mặc dù bộ gen người vô cùng đồ sộ, việc thay đổi sai lệch ở duy nhất một vị trí cũng hoàn toàn đủ khả năng làm thay đổi toàn bộ cấu trúc và chức năng của gen đó.


Trong nhiều tài liệu nghiên cứu trước đây, sự thay đổi của một nucleotide thường bị gọi lẫn lộn và đánh đồng với khái niệm "Đa hình nucleotide đơn" (Single Nucleotide Polymorphism - SNP). Tuy nhiên, trong phân tích tin sinh học và di truyền lâm sàng hiện đại, hai khái niệm này có sự phân định rõ ràng:

- Đa hình đơn nucleotide (SNP): Khái niệm này không chỉ mô tả kiểu thay đổi cấu trúc mà còn bị ràng buộc bởi điều kiện tần suất. Một sự thay đổi chỉ được công nhận là SNP khi nó xuất hiện phổ biến trong quần thể (tần suất bắt buộc từ 1% trở lên). Do phổ biến, các SNP thường là những biến dị mang tính tiến hóa tự nhiên, quy định sự đa dạng sinh học (như màu da, màu mắt) và hiếm khi là nguyên nhân trực tiếp gây ra bệnh lý.
- Đột biến đơn nucleotide (SNV): Khái niệm này là một thuật ngữ trung lập tuyệt đối. Nó chỉ đơn thuần mô tả: có một nucleotide bị thay đổi tại một tọa độ nhất định trên chuỗi DNA, bất kể tần suất xuất hiện của đột biến đó là cực kỳ hiếm (ví dụ 0.0001% - chỉ gặp ở một vài bệnh nhân) hay rất phổ biến.

Chính vì sự khác biệt này, phần lớn các đột biến nguy hiểm gây ra các bệnh di truyền hiếm gặp hoặc ung thư đều là các SNV (do chúng có tần suất rất thấp trong quần thể).

### Tác động của đột biến lên chức năng Protein

Dựa trên nguyên lý trung tâm của sinh học phân tử, luồng thông tin di truyền từ DNA sẽ được phiên mã (transcription) thành phân tử RNA thông tin (mRNA), và sau đó tiếp tục được dịch mã (translation) tại ribosome để tổng hợp nên các chuỗi polypeptide cấu tạo thành protein. Quá trình dịch mã hoạt động dựa trên nguyên tắc bộ ba mã di truyền (codon), trong đó cứ ba nucleotide liên tiếp sẽ quy định một axit amin cụ thể. Tùy thuộc vào vị trí đột biến (nằm trong vùng mã hóa hay vùng điều hòa) và bản chất của nucleotide bị thay thế, đột biến điểm có thể gây ra các tác động ở những mức độ khác nhau,


Đột biến được chia làm ba nhóm chính:

- Đột biến câm (Silent Mutation): Do tính thoái hóa của mã di truyền (một axit amin có thể được mã hóa bởi nhiều codon khác nhau), sự thay đổi nucleotide thường ở vị trí thứ ba của codon tạo ra một bộ ba mã hóa mới nhưng vẫn dịch mã ra cùng một loại axit amin như ban đầu. Kết quả là cấu trúc bậc một của protein không bị thay đổi. Loại đột biến này thường không làm suy giảm chức năng protein và phần lớn được phân loại là lành tính (Benign).
- Đột biến sai nghĩa (Missense Mutation): Việc thay thế nucleotide làm cho codon ban đầu biến thành một codon mới, từ đó mã hóa cho một loại axit amin hoàn toàn khác. Hậu quả của sự thay đổi này phụ thuộc rất lớn vào tính chất hóa lý của axit amin mới thay thế. Điều này có thể làm thay đổi cấu trúc gấp khúc không gian 3D của protein, làm biến dạng trung tâm hoạt động của enzyme, dẫn đến việc giảm sút hoặc mất hoàn toàn chức năng sinh học bình thường của protein.
- Đột biến vô nghĩa (Nonsense Mutation): Đây là trường hợp đột biến tạo ra một trong ba bộ ba kết thúc sớm (Stop codon). Sự xuất hiện đột ngột của tín hiệu dừng này khiến quá trình dịch mã bị gián đoạn giữa chừng. Chuỗi polypeptide tạo ra bị cắt cụt, không thể cuộn gập thành cấu trúc hoàn chỉnh và thường bị tế bào phân hủy. Sự thiếu hụt các protein chức năng này là nguyên nhân trực tiếp gây ra các hội chứng bệnh lý di truyền nghiêm trọng (Pathogenic).

## Các cơ sở dữ liệu tin sinh học

Bài toán phân loại tính gây bệnh của đột biến gen bằng phương pháp học sâu đòi hỏi một lượng lớn dữ liệu có độ tin cậy cao. Điều này yêu cầu sự kết hợp chặt chẽ giữa dữ liệu biến dị đã được kiểm chứng lâm sàng và dữ liệu giải trình tự thông lượng cao được ánh xạ trên một hệ gen tham chiếu chuẩn. Hai nền tảng dữ liệu cốt lõi phục vụ cho mục đích này là ClinVar và GRCh38.

### Cơ sở dữ liệu ClinVar (NCBI)

ClinVar là một kho lưu trữ dữ liệu y sinh học công khai, quy mô lớn được vận hành và duy trì bởi Trung tâm Thông tin Công nghệ sinh học Quốc gia Hoa Kỳ (NCBI). Mục tiêu cốt lõi của cơ sở dữ liệu này là tập hợp, chuẩn hóa và cung cấp các báo cáo về mối quan hệ giữa các biến dị di truyền ở người và các kiểu hình hoặc tình trạng bệnh lý tương ứng.

Trong kiến trúc của hệ thống dự đoán, ClinVar đóng vai trò tối quan trọng là cung cấp tập dữ liệu gán nhãn để huấn luyện mô hình học sâu. Các biến dị trong ClinVar được thu thập từ nhiều nguồn uy tín như các phòng thí nghiệm chẩn đoán lâm sàng, các viện nghiên cứu và các kho dữ liệu chuyên sâu. Chúng được đánh giá bởi các chuyên gia y tế theo tiêu chuẩn của Hiệp hội Di truyền Y học Hoa Kỳ (ACMG) và phân loại theo các mức độ ý nghĩa lâm sàng (Clinical Significance), cụ thể bao gồm 5 phân nhóm chính:

- Pathogenic (Gây bệnh): Biến dị có bằng chứng rõ ràng gây ra bệnh lý.
- Likely Pathogenic (Có khả năng gây bệnh): Biến dị có khả năng cao gây bệnh nhưng chưa đủ bằng chứng tuyệt đối.
- Benign (Lành tính): Biến dị không gây ảnh hưởng đến chức năng sinh học hoặc không gây bệnh.
- Likely Benign (Có khả năng lành tính): Biến dị mang đặc tính không gây bệnh cao.
- Uncertain Significance (VUS - Biến dị chưa rõ ý nghĩa): Biến dị chưa có đủ bằng chứng lâm sàng để kết luận là lành tính hay gây bệnh.

Bên cạnh nhãn phân loại, ClinVar còn cung cấp một hệ thống đánh giá (Review Status) để thể hiện mức độ đồng thuận và độ tin cậy của dữ liệu. Việc tận dụng hệ thống Review Status này cho phép các nhà nghiên cứu lọc bỏ các dữ liệu nhiễu, mâu thuẫn, từ đó xây dựng được một tập dữ liệu sạch và chất lượng cao để tối ưu hóa quá trình huấn luyện mô hình học sâu.

### Hệ gen tham chiếu GRCh38

Hệ gen tham chiếu của người (Human Reference Genome) là một cơ sở dữ liệu kỹ thuật số đại diện cho toàn bộ chuỗi DNA chuẩn của con người. Dữ liệu này được lắp ráp, duy trì và cập nhật liên tục bởi tổ chức Genome Reference Consortium (GRC). Phiên bản GRCh38 (hay hg38) là bản lắp ráp tiên tiến và hoàn thiện nhất hiện nay, đóng vai trò như một "hệ tọa độ gốc" chuẩn cho toàn bộ các nghiên cứu tin sinh học trên toàn cầu.

Sự cần thiết của GRCh38 trong bài toán dự đoán đột biến đến từ đặc thù của dữ liệu biến dị. Bản thân các file dữ liệu thô tải về từ ClinVar chỉ cung cấp thông tin ở mức độ điểm: tọa độ nhiễm sắc thể xảy ra đột biến, chữ cái nucleotide gốc (Reference allele) và chữ cái bị biến đổi (Alternate allele). Nó hoàn toàn thiếu đi bối cảnh chuỗi xung quanh.

Trong khi đó, các mô hình học sâu như mạng thần kinh tích chập 1D (1D-CNN) lại hoạt động dựa trên việc nhận diện các mẫu và trích xuất đặc trưng không gian cục bộ. Bằng cách sử dụng cơ sở dữ liệu GRCh38 thông qua kho RefSeq của NCBI, hệ thống có thể truy vấn dựa trên tọa độ của ClinVar để trích xuất một đoạn trình tự DNA ngữ cảnh (ví dụ: một cửa sổ trượt chứa 500 đến 1000 base pairs phân bố đối xứng xung quanh điểm đột biến). Đoạn trình tự này mang các thông tin thiết yếu về các motif liên kết, mức độ bảo tồn tiến hóa và cấu trúc DNA cục bộ.

Đặc biệt, việc đảm bảo tính đồng bộ nghiêm ngặt về phiên bản lắp ráp là yếu tố sống còn. Cả tọa độ biến dị từ ClinVar và trình tự trích xuất từ cơ sở dữ liệu phải cùng tuân theo hệ tọa độ của GRCh38. Nếu xảy ra sự sai lệch phiên bản (ví dụ tọa độ ClinVar dùng GRCh37 nhưng trích xuất trên GRCh38), toàn bộ tập dữ liệu đầu vào sẽ bị xô lệch tọa độ, khiến mô hình học sâu học sai đặc trưng và thất bại hoàn toàn trong việc dự đoán.

## Mạng thần kinh tích chập một chiều (1D-CNN)

Trong vài thập kỷ qua, học sâu (Deep Learning) đã chứng tỏ mình là một công cụ mạnh mẽ trong việc nhận dạng các mẫu phức tạp trong dữ liệu lớn. Đối với bài toán phân tích chuỗi DNA để dự đoán tính gây bệnh của biến dị di truyền, mạng thần kinh tích chập một chiều (1D-CNN) thể hiện sự ưu việt vượt trội so với các thuật toán học máy truyền thống (như SVM hay Random Forest). Thay vì phụ thuộc vào việc kỹ sư phải trích xuất đặc trưng thủ công (ví dụ: đếm tần suất k-mer), 1D-CNN có khả năng tự động học biểu diễn dữ liệu trực tiếp từ trình tự nucleotide thô.

### Giới thiệu chung về 1D-CNN trong phân tích chuỗi

Khác với kiến trúc mạng 2D-CNN thường được sử dụng trong xử lý hình ảnh y tế hay thị giác máy tính (nơi các bộ lọc di chuyển theo không gian hai chiều chiều dài và chiều rộng), 1D-CNN được thiết kế chuyên biệt để xử lý các loại dữ liệu có tính chuỗi và không gian một chiều, điển hình là chuỗi thời gian hoặc trình tự ký tự sinh học (DNA/RNA/Protein).

Trong bài toán này, đầu vào của mạng 1D-CNN không phải là một chuỗi ký tự dạng văn bản đơn thuần, mà là một ma trận số học được chuyển đổi thông qua phương pháp mã hóa One-hot (One-hot encoding). Cụ thể, một chuỗi DNA có độ dài L sẽ được biểu diễn dưới dạng ma trận , trong đó 4 tương ứng với 4 loại nucleotide (A, C, G, T). Thông qua các phép toán tích chập trượt dọc theo chiều dài L của chuỗi, mạng 1D-CNN đóng vai trò như một cỗ máy dò tìm, có khả năng tự động phát hiện các motif sinh học (các mẫu trình tự DNA ngắn mang chức năng cụ thể hoặc có liên quan trực tiếp đến cơ chế gây bệnh của đột biến).

### Các thành phần cốt lõi của 1D-CNN

Để xây dựng một mô hình có khả năng suy luận trên chuỗi gen, kiến trúc 1D-CNN được cấu thành từ các lớp chồng lên nhau theo thứ bậc, bao gồm:

- Lớp tích chập (Convolution layer):
  - Lớp tích chập là thành phần cơ bản và quan trọng nhất của kiến trúc CNN, thực hiện nhiệm vụ trích xuất đặc trưng. Cơ chế hoạt động dựa trên một tập hợp các ma trận trọng số nhỏ, được gọi là bộ lọc (kernel/filter), trượt qua mảng trình tự đầu vào. Tại mỗi vị trí cửa sổ trượt, hệ thống thực hiện phép tính tổng của tích từng phần tử giữa bộ lọc và đoạn đầu vào để tạo ra một bản đồ đặc trưng.
  - Về mặt toán học, phép tích chập một chiều tại vị trí i với bộ lọc W (kích thước k) và chuỗi đầu vào X được biểu diễn như sau:


(Trong đó Y_i là giá trị đầu ra tại vị trí i, b là độ lệch bias).

Các đặc tính ưu việt và siêu tham số của lớp này bao gồm:

- - Kích thước bộ lọc (Kernel size - k): Quyết định độ dài của đoạn motif sinh học mà mạng cần tìm kiếm (ví dụ: k=8 tương đương việc quét các motif dài 8 nucleotide).
    - Số lượng bộ lọc (Number of filters): Quyết định số lượng motif khác nhau mà mô hình có thể học được đồng thời.
    - Chia sẻ trọng số (Weight sharing) và Bất biến dịch (Translation invariance): Cùng một bộ trọng số W được sử dụng để quét qua toàn bộ chuỗi. Điều này giúp giảm thiểu hàng triệu tham số tính toán so với mạng nơ-ron truyền thống, đồng thời mang ý nghĩa sinh học cốt lõi: một motif liên quan đến đột biến gây bệnh có thể được mô hình nhận diện chính xác bất kể nó nằm ở vị trí nào (đầu, giữa hay cuối) trên đoạn DNA ngữ cảnh.
- Hàm kích hoạt (Activation Function): Đầu ra Y của phép toán tuyến tính từ lớp tích chập sẽ được đưa qua một hàm kích hoạt để phá vỡ tính tuyến tính, giúp mạng có khả năng biểu diễn các không gian đặc trưng sinh học phức tạp. Hàm kích hoạt phổ biến và hiệu quả nhất hiện nay cho kiến trúc này là ReLU (Rectified Linear Unit), có phương trình:


Hàm ReLU biến đổi tất cả các giá trị âm thành 0 và giữ nguyên các giá trị dương. Trong sinh học tính toán, điều này tạo ra tính thưa cho mạng: mô hình chỉ "kích hoạt" tín hiệu khi bộ lọc thực sự tìm thấy motif phù hợp, và bỏ qua các vùng trình tự không mang thông tin. Đồng thời, ReLU giúp khắc phục triệt để hiện tượng triệt tiêu đạo hàm (vanishing gradient), cho phép huấn luyện các mạng sâu hơn.

- Lớp tổng hợp (Pooling layer): Lớp tổng hợp được đặt ngay sau lớp tích chập, thực hiện việc giảm mẫu để nén kích thước không gian của các bản đồ đặc trưng. Phương pháp phổ biến nhất trong phân tích chuỗi là Max Pooling, trong đó hệ thống sẽ sử dụng một cửa sổ trượt và chỉ giữ lại giá trị lớn nhất trong cửa sổ đó. Trong ngữ cảnh phân tích đột biến gen, Max Pooling mang một ý nghĩa sinh lý học sâu sắc: Mô hình học sâu không cần quan tâm đến tọa độ chính xác tuyệt đối của một motif chức năng, mà chỉ cần xác nhận "sự tồn tại" của tín hiệu mạnh nhất đại diện cho motif đó trong một vùng lân cận. Cơ chế này tăng cường tính bất biến dịch cục bộ (local translation invariance) và giúp mô hình hạn chế hiện tượng học vẹt.
- Lớp kết nối đầy đủ (Fully connected layer): Sau khi luồng dữ liệu truyền qua nhiều khối Convolution và Pooling, các bản đồ đặc trưng cấp cao sẽ được "duỗi phẳng" từ ma trận thành một vector một chiều đơn nhất. Vector này được đưa vào các mạng nơ-ron truyền thống (Fully Connected Network). Ở lớp này, mỗi nơ-ron được liên kết mật thiết với tất cả các nơ-ron của lớp trước đó. Lớp kết nối đầy đủ đóng vai trò như một bộ phân loại tổng hợp. Nó thu thập sự hiện diện của các motif sinh học khác nhau (từ phần trích xuất đặc trưng) để đưa ra quyết định cuối cùng. Do mục tiêu của dự án là phân loại nhị phân (Dự đoán một đột biến là _Lành tính - Benign_ hay _Gây bệnh - Pathogenic_), nơ-ron ở lớp đầu ra sẽ được áp dụng hàm kích hoạt Sigmoid:


Hàm Sigmoid làm nhiệm vụ nén giá trị đầu ra về một số thực liên tục trong khoảng (0, 1). Giá trị này biểu diễn trực tiếp xác suất phân lớp P(y=1 | X), giúp hệ thống đánh giá được mức độ tin cậy khi dự đoán tính gây bệnh của một biến dị di truyền.

# Phương pháp dự đoán tính gây hại của đột biến gen thông qua CNN

Trong chương này, báo cáo trình bày chi tiết phương pháp xây dựng mô hình học sâu sử dụng mạng nơ-ron tích chập một chiều (1D - CNN), để dự đoán tính gây hại của đột biến gen đơn điểm. Bài toán được đặt dưới dạng phân loại nhị phân: với mỗi đột biến đơn nucleotide, mô hình cần dự đoán đột biến đó thuộc nhóm gây bệnh hay lành tính dựa trên chuỗi DNA xung quanh vị trí đột biến.

Khác với các bài toán phân loại ảnh, dữ liệu đầu vào của bài toán này là chuỗi sinh học gồm các ký tự nucleotide Adenine (A), Cytosine (C), Guanine (G), và Thymine (T). Mỗi đột biến đơn điểm không chỉ được mô tả bởi nucleotide ban đầu và nucleotide thay thế, mà còn chịu ảnh hưởng bởi ngữ cảnh chuỗi DNA xung quanh. Vì vậy, thay vì chỉ đưa vào mô hình một cặp biến đổi dạng REF -> ALT, dự án xây dựng đầu vào là một cửa sổ DNA có độ dài cố định bao quanh vị trí đột biến. Trên cửa sổ đó, các bộ lọc tích chập của mạng nơ-ron có thể học các motif sinh học, mẫu nucleotide cục bộ, tín hiệu hàm lượng GC, tín hiệu gần tâm đột biến và các đặc trưng xa hơn.

## Dữ liệu

### Nguồn dữ liệu

Dữ liệu chính của dự án được lấy từ cơ sở dữ liệu ClinVar, cụ thể là file variant_summary.txt. Đây là file tổng hợp các biến thể di truyền đã được ClinVar ghi nhận, bao gồm thông tin về vị trí đột biến, loại đột biến, allele tham chiếu, allele thay thế, ý nghĩa lâm sàng và mức độ đánh giá của biến thể. Bên cạnh ClinVar, dự án sử dụng hệ gen tham chiếu người GRCh38 ở định dạng FASTA để trích xuất chuỗi DNA thật quanh vị trí đột biến. ClinVar cung cấp nhãn lâm sàng, chẳng hạn như Benign, Likely benign, Pathogenic, Likely pathogenic, còn GRCh38 cung cấp chuỗi DNA tham chiếu tại từng vị trí trên hệ gen.

Dữ liệu đầu vào ban đầu bao gồm nhiều loại biến thể khác nhau. Tuy nhiên, trong phạm vi dự án này, mô hình được giới hạn vào bài toán: dự đoán tính gây hại của đột biến đơn nucleotide, tức single nucleotide variant.

### Lọc dữ liệu

Quá trình lọc dữ liệu được thực hiện nhằm bảo đảm mỗi mẫu đưa vào mô hình có tọa độ chính xác, nhãn đáng tin cậy và chuỗi DNA hợp lệ. Các điều kiện lọc chính gồm:

Thứ nhất, nhãn của mô hình được định nghĩa như sau:

| Nhóm ClinVar      | Nhãn mô hình | Ý nghĩa                        |
| ----------------- | ------------ | ------------------------------ |
| Benign            | 0            | Biến thể lành tính             |
| ---               | ---          | ---                            |
| Likely benign     | 0            | Biến thể có khả năng lành tính |
| ---               | ---          | ---                            |
| Pathogenic        | 1            | Biến thể gây hại               |
| ---               | ---          | ---                            |
| Likely pathogenic | 1            | Biến thể có khả năng gây hại   |
| ---               | ---          | ---                            |

Các nhãn không chắc chắn như Uncertain significance, Conflicting classifications, drug response, risk factor không được sử dụng trong quá trình huấn luyện chính. Mô hình học sâu rất nhạy với nhiễu nhãn, nếu đưa các biến thể chưa rõ ý nghĩa vào tập huấn luyện, mô hình có thể học sai.

Thứ hai, chỉ giữ các đột biến thuộc hệ gen tham chiếu GRCh38. Điều này giúp tránh sai lệch tọa độ giữa các phiên bản hệ gen khác nhau (VD: GRCh37).

Thứ ba, chỉ giữ các biến thể có Type == single nucleotide variant (SNV). Với deletion hoặc insertion dài, chiều dài REF và ALT không còn bằng một base, khi đó cách biểu diễn sparse ALT tại một vị trí trung tâm sẽ không còn phù hợp.

Thứ tư, chỉ giữ các biến thể có ReferenceAlleleVCF và AlternateAlleleVCF là một trong bốn base chuẩn A, C, G, T. Các trường hợp chứa ký tự ngoài bốn base này bị loại bỏ vì không thể mã hóa one-hot một cách rõ ràng.

Thứ năm, allele tham chiếu trong ClinVar được đối chiếu lại với FASTA GRCh38. Nếu base tại vị trí tương ứng trong FASTA không khớp với ReferenceAlleleVCF, mẫu đó bị loại bỏ. Đây là bước kiểm tra rất quan trọng vì sai lệch giữa bảng biến thể và hệ gen tham chiếu có thể khiến mô hình học sai ngữ cảnh.

Thứ bảy, các biến thể quá gần đầu hoặc cuối contig, không đủ độ dài cửa sổ 601 bp, cũng bị loại bỏ. Ngoài ra, các cửa sổ chứa ký tự không thuộc A, C, G, T cũng không được giữ lại.

Sau khi chuẩn hóa, tập dữ liệu có khoảng khoảng 1.426 triệu SNV có allele tham chiếu khớp với FASTA.

### Phương pháp mã hóa chuỗi DNA (Biểu diễn ma trận Tensor Sparse ALT 8-Channel)

Mạng nơ-ron tích chập không thể tiếp nhận trực tiếp các chuỗi ký tự văn bản (A, C, G, T). Để chuyển đổi ngữ cảnh sinh học bao quanh đột biến thành dữ liệu, hệ thống thiết lập một cửa sổ trượt có độ dài cố định là 601 bp. Cấu trúc này bao gồm: 300 bp thuộc vùng bên trái, 1 bp cho vị trí nucleotide đột biến, và 300 bp thuộc vùng phía bên phải.

Chuỗi ký tự DNA trong phạm vi cửa sổ 601 bp được số hóa thành một Tensor đầu vào có kích thước là (601, 8), tương ứng với 601 vị trí và 8 kênh đặc trưng chuyên biệt. Không gian 8 kênh này được phân tách theo thiết kế Sparse ALT 8-channel nhằm loại bỏ sự lặp lại dư thừa dữ liệu:

- Kênh 1 đến Kênh 4 (Kênh chuỗi tham chiếu - REF): Biểu diễn toàn bộ trình tự bối cảnh của mạch DNA gốc dựa trên hệ gen tham chiếu GRCh38 trải dài suốt chiều dài cửa sổ 601 bp thông qua mã hóa One-hot tiêu chuẩn, trong đó mỗi nucleotide là một vector nhị phân chuẩn hóa: A=\[1,0,0,0\], C=\[0,1,0,0\], G=\[0,0,1,0\], T=\[0,0,0,1\].
- Kênh 5 đến Kênh 8 (Kênh đột biến thưa - Sparse ALT): Biểu diễn tín hiệu của base thay thế. Tận dụng đặc tính của đột biến đơn điểm (SNV) - trình tự chuỗi ALT trùng khớp với chuỗi REF tại mọi tọa độ ngoại trừ điểm xảy ra đột biến - hệ thống áp dụng cơ chế kích hoạt thưa. Bốn kênh này chỉ kích hoạt giá trị mã One-hot tương ứng với base đột biến tại vị trí trung tâm (tọa độ thứ 301). Tại 600 vị trí còn lại bao quanh 2 đầu cửa sổ, giá trị của 4 kênh này được thiết lập bằng 0.


Thiết kế này mang lại các lợi ích:

- Dung hợp thông tin sớm: Ép các bộ lọc (kernels) tích chập ở tầng kiến trúc đầu tiên tiếp nhận đồng thời cặp tín hiệu đối chiếu chuyển đổi (REF => ALT) tại đúng tâm đột biến, giúp mô hình học trực tiếp các liên kết motif phụ thuộc ngữ cảnh .
- Định vị không gian ẩn: Sự xuất hiện duy nhất của giá trị khác 0 tại tọa độ 301 đóng vai trò đánh dấu vị trí đột biến cho mạng CNN mà không cần tiêu tốn tài nguyên cho một kênh chỉ mục độc lập.
- Tối ưu hóa tài nguyên: Loại bỏ việc lặp lại chuỗi ký tự ALT, làm giảm dung lượng bộ nhớ lưu trữ, tăng tốc độ đọc nạp từ ổ đĩa và giải phóng áp lực RAM trên các môi trường tính toán.

### Chiến lược chống rò rỉ dữ liệu

Trong các bài toán học máy thông thường, dữ liệu thường được chia ngẫu nhiên thành tập huấn luyện (Train), tập xác thực (Validation) và tập kiểm thử (Test). Tuy nhiên, với dữ liệu hệ gen, chia ngẫu nhiên có thể dẫn đến rò rỉ dữ liệu. Nguyên nhân là các biến thể nằm gần nhau trên hệ gen hoặc nằm trong cùng một gen có thể có chuỗi ngữ cảnh rất giống nhau. Nếu một biến thể xuất hiện trong tập train và một biến thể rất gần nó xuất hiện trong tập test, mô hình có thể học thuộc vùng DNA thay vì học quy luật tổng quát về tác động của đột biến.

Do đó, để giải quyết vấn đề này dữ liệu được chia bằng chiến thuật nghiêm ngặt dựa trên các block hệ gen, kết hợp với loại bỏ các mẫu quá gần nhau giữa các split.

- Genome_block split: Toàn bộ hệ gen được phân thành các khối cố định có kích thước 1.000.000 bp. Việc phân chia các tập dữ liệu Huấn luyện (Train), Xác thực (Validation) và Kiểm thử (Test) được thực hiện rạch ròi theo ranh giới của các khối block này, đảm bảo không có sự chồng lấn không gian.
- Coordinate Purge: Hệ thống tiến hành rà soát khoảng cách vật lý, lọc và loại bỏ đột biến nào thuộc tập Validation hoặc Test nếu tọa độ của chúng nằm quá gần tập Train dưới một khoảng cách an toàn là 5.000 bp.
- Exact REF Sequence Purge: Rà soát chuỗi văn bản, loại bỏ các mẫu có trình tự bối cảnh mạch tham chiếu (REF) trùng lặp giữa các phân đoạn split.

Sau khi áp dụng bộ lọc này, tập dữ liệu thu được bao gồm:

| Phân đoạn dữ liệu     | Tổng số mẫu | Benign/Likely benign | Pathogenic/Likely pathogenic |
| --------------------- | ----------- | -------------------- | ---------------------------- |
| Huấn luyện (Train)    | 1,002,376   | 883,610              | 118,766                      |
| ---                   | ---         | ---                  | ---                          |
| Xác thực (Validation) | 205,864     | 180,524              | 25,340                       |
| ---                   | ---         | ---                  | ---                          |
| Kiểm thử (Test)       | 216,351     | 192,330              | 24,021                       |
| ---                   | ---         | ---                  | ---                          |

Sau khi thực hiện quy trình lọc này, khoảng cách tối thiểu từ mỗi mẫu thuộc tập Validation tới tập Train đạt 5.001 bp. Đặc biệt, khoảng cách vật lý trung vị từ tập Test tới không gian dữ liệu đã biết (Train/Validation) đạt tới 388.904 bp. Chỉ số khoảng cách lớn này là minh chứng kỹ thuật khẳng định mô hình không thể gian lận dữ liệu.

Bộ dữ liệu sau khi làm sạch mang đặc tính mất cân bằng lớp rất lớn, phản ánh đúng quy luật phân bố trong tự nhiên: tần suất xuất hiện các biến dị lành tính luôn áp đảo các đột biến gây hại. Qua thống kê, tỷ lệ các mẫu Gây hại chỉ chiếm vỏn vẹn 11.8% tổng quy mô dữ liệu. Sự chênh lệch này mang lại một hệ quả đáng lưu ý: Nếu dự án sử dụng thang đo Độ chính xác (Accuracy) làm thước đo chính, mô hình chỉ cần áp dụng một chiến lược là dự đoán tất cả mọi mẫu đầu vào đều là Lành tính thì điểm Accuracy vẫn đạt mức tối ưu là 88.2%. Vì vậy, trong bài toán này, accuracy không được xem là chỉ số đánh giá chính. Các chỉ số quan trọng hơn gồm: PR-AUC, ROC-AUC, precision của lớp pathogenic, recall của lớp pathogenic, F1-score của lớp pathogenic, confusion matrix.

### Kiểm chứng theo thời gian (Temporal Validation)

Phép chia genome-block ở trên chỉ kiểm soát rò rỉ về mặt **không gian** (các biến thể gần nhau hoặc cùng vùng gen). Tuy nhiên, dữ liệu ClinVar còn mang một chiều thiên lệch khác là **thời gian**: cơ sở dữ liệu liên tục được mở rộng và nhãn được tinh chỉnh qua từng năm. Mỗi giai đoạn, cộng đồng nghiên cứu lại tập trung vào một số vùng gen nhất định, khiến các biến thể được gán nhãn trong cùng một giai đoạn có xu hướng cụm vào cùng vài vùng. Nếu chia ngẫu nhiên hoặc chỉ chia theo gen, các mẫu của một vùng đang được nghiên cứu mạnh sẽ xuất hiện ở cả tập huấn luyện lẫn kiểm thử; khi đó mô hình ghi điểm nhờ đã "thấy trước" loại pattern đó chứ không phải nhờ hiểu tác động đột biến. Đây là một dạng rò rỉ mà phép chia theo không gian không phát hiện được.

Có thể hình dung bằng phép so sánh với mạng CNN nhận dạng ảnh: nếu trước năm 2022 dữ liệu chỉ chứa "ảnh mèo, gà" (các vùng gen, motif đã được nghiên cứu tới thời điểm đó), thì một phép kiểm tra không rò rỉ phải đưa "ảnh chó" — những vùng gen, motif mới chỉ xuất hiện sau 2022 — vào riêng tập kiểm thử. Phép chia ngẫu nhiên hoặc chỉ theo gen lại để "ảnh chó" lọt vào cả train lẫn test: mô hình đã thấy loại pattern đó lúc học nên đoán đúng không chứng minh được năng lực thật.

Để mô phỏng đúng tình huống triển khai thực tế ("huấn luyện trên kiến thức quá khứ, dự đoán biến thể mới trong tương lai"), dự án bổ sung chiến lược phân tách theo thời gian dựa trên hai phiên bản ClinVar khác nhau:

- **Tập huấn luyện / xác thực**: các biến thể đã tồn tại trong phiên bản ClinVar 2022-12.
- **Tập kiểm thử**: các biến thể **chỉ xuất hiện** trong phiên bản 2026-05 (hoàn toàn mới so với mốc 2022).

Mốc 2022-12 được chọn không phải tùy ý mà nhằm cân bằng kích thước hai tập: train đủ lớn để học và test đủ lớn để đánh giá tin cậy. Tập xác thực được tách ra từ chính dữ liệu ≤2022 theo genome-block (region-disjoint với train) và chỉ dùng để chọn mô hình cùng ngưỡng quyết định, hoàn toàn không chạm vào tập kiểm thử tương lai.

| Phân đoạn dữ liệu | Tổng số mẫu | Cách lấy |
| --- | --- | --- |
| Huấn luyện (Train) | 610.407 | Biến thể ≤ 2022-12 (đã tách validation) |
| Xác thực (Validation) | 102.874 | Tách từ ≤ 2022-12 theo genome-block |
| Kiểm thử (Test) | 586.778 | Biến thể mới (chỉ có trong 2026-05) |

Tỉ lệ train/test ở đây xấp xỉ 50/50, khác với quy ước 80/20 thường gặp. Đây là **hệ quả tự nhiên** của phép chia theo thời gian chứ không phải lựa chọn thủ công: số biến thể ClinVar được gán nhãn trong giai đoạn 2022-2026 xấp xỉ lượng tích lũy trước đó. Một tập kiểm thử lớn (586.778 mẫu) thực chất làm cho kết quả đánh giá ổn định và đáng tin hơn.

## Kiến trúc mô hình mạng tích chập Dilated 1D-CNN và Cơ chế đối ngẫu

Để phân tích và trích xuất hiệu quả các đặc trưng sinh học từ chuỗi DNA đầu vào (với cửa sổ 601 bp), mô hình đòi hỏi một kiến trúc mạng nơron đủ sâu để nắm bắt ngữ cảnh rộng lớn, nhưng đồng thời không được đánh mất thông tin về tọa độ không gian chính xác của đột biến. Do đó, dự án ứng dụng kiến trúc mạng tích chập một chiều giãn nở (Dilated 1D-CNN) kết hợp cùng kỹ thuật học máy đối ngẫu dựa trên cấu trúc mạch đôi của DNA.

### Kiến trúc các lớp tích chập một chiều giãn nở

Một trong những nhược điểm của mạng CNN truyền thống khi xử lý chuỗi trình tự là việc lạm dụng các lớp gộp (Pooling layer). Thao tác giảm mẫu này làm mất đi chi tiết vị trí không gian. Đối với bài toán phân loại SNV, vị trí trung tâm (nơi xảy ra đột biến) chứa tín hiệu cốt lõi, việc làm mờ vị trí này sẽ làm suy giảm nghiêm trọng độ chính xác của mô hình.

Để khắc phục nhược điểm này, dự án triển khai kiến trúc Dilated 1D CNN. Thay vì giảm độ phân giải không gian qua các lớp Pooling, Dilated CNN mở rộng trường nhìn bằng cách gia tăng khoảng cách lấy mẫu bên trong chính các kernel tích chập. Cụ thể, mô hình thiết lập chuỗi các lớp tích chập với hệ số giãn nở tăng dần theo cấp số nhân: 1, 2, 4, 8, 16, 32 và 64.

Thiết kế phân tầng giãn nở này mang lại những ưu thế vượt trội trong việc phân tích motif DNA:

- Học đặc trưng cục bộ: Các lớp tích chập ở giai đoạn đầu với hệ số giãn nở nhỏ (1, 2, 4) đảm nhiệm việc quét và học các motif cục bộ nằm ngay sát vị trí đột biến.
- Học ngữ cảnh vĩ mô: Các lớp tiến sâu vào mạng với hệ số giãn nở lớn (8 đến 64) cho phép kernel vươn tầm nhìn ra xa để học các pattern phức tạp hơn. Các pattern này bao gồm các vùng bảo tồn di truyền, motif lặp, ngữ cảnh GC, hoặc cấu trúc sequence rộng hơn.
- Bảo toàn tọa độ không gian: Nhờ cơ chế này, mô hình có thể nhìn được khoảng cách xa hơn mà không cần giảm độ phân giải quá sớm. Nhờ đó, đột biến nằm ở vị trí trung tâm của chuỗi 601 bp luôn giữ được định vị chính xác xuyên suốt quá trình lan truyền tiến.

Sau khi thông tin đã được lan truyền đủ xa qua hàng loạt các lớp tích chập giãn nở, đặc trưng của chuỗi trình tự được tổng hợp lại thông qua các lớp Global Max Pooling và Global Mean Pooling. Bước này giúp gom các đặc trưng quan trọng nhất trên toàn bộ chiều dài cửa sổ 601 bp để chuyển hóa thành một vector phẳng, sẵn sàng đưa vào bộ phân loại ở các lớp kết nối đầy đủ cuối cùng.

### Khai thác tính bổ sung ngược của DNA (RC Augment & RC TTA)

Bên cạnh kiến trúc mạng nơron, dự án còn tối ưu hóa luồng dữ liệu bằng cách lồng ghép trực tiếp bản chất vật lý của phân tử DNA vào quá trình học. Trong tự nhiên, phân tử DNA là một chuỗi xoắn kép gồm hai mạch chạy ngược chiều nhau và đối xứng theo nguyên tắc bổ sung: Adenine (A) liên kết với Thymine (T), Cytosine (C) liên kết với Guanine (G). Do đó, cùng một đột biến gen hoàn toàn có thể được đọc từ mạch bên này (mạch xuôi) hoặc mạch bên kia (mạch ngược) mà bản chất gây bệnh của nó không hề thay đổi.

Tuy nhiên, file dữ liệu FASTA chỉ cung cấp một chiều đọc duy nhất. Nếu không cẩn thận, AI sẽ có thể học vẹt chiều đọc. Để triệt tiêu rủi ro này, hệ thống áp dụng cơ chế học đối ngẫu song hành:

- Pha Huấn luyện (Reverse-Complement Augmentation - RC Augment): Trong quá trình huấn luyện AI, luồng cấp dữ liệu sẽ ngẫu nhiên đưa vào mạng nơ-ron trình tự gốc hoặc trình tự bổ sung của nó. Kỹ thuật này đóng vai trò như một bộ điều chuẩn mạnh mẽ, giúp giảm thiểu hiện tượng mô hình học vẹt chiều đọc cố định của file tham chiếu FASTA. Nó ép mạng CNN phải học cách nhận diện các motif sinh học tương đương bất kể chúng nằm trên mạch DNA nào.
- Pha chẩn đoán (Reverse-Complement Test-Time Augmentation - RC TTA): Khi thực hiện dự đoán (evaluate/test) trên một mẫu đột biến mới, hệ thống sẽ chạy dự đoán đồng thời trên cả mạch gốc và mạch bổ sung, sau đó lấy giá trị trung bình của hai kết quả này. Phương pháp TTA giúp các dự đoán trở nên cực kỳ ổn định, đặc biệt loại bỏ hoàn toàn các sai số nếu AI vô tình bị nhạy cảm với một chiều đọc văn bản cụ thể. Cải tiến này đã giúp nâng đáng kể chỉ số độ chính xác diện rộng (PR-AUC) của toàn hệ thống.

## Huấn luyện, Chiến lược xử lý mất cân bằng lớp và Kết quả thực nghiệm

### Chiến lược xử lý mất cân bằng phân phối lớp và Ổn định huấn luyện

Đặc thù lớn nhất của tập dữ liệu đột biến sinh học ClinVar là sự mất cân bằng lớp cực kỳ nghiêm trọng. Trong tự nhiên, số lượng đột biến không gây hại luôn áp đảo các đột biến gây bệnh. Theo phân tích phân phối dữ liệu, tỷ lệ các đột biến mang tính chất Gây bệnh (Pathogenic/Likely pathogenic) chỉ chiếm khoảng 11,8% tổng quy mô tập dữ liệu, trong khi phần lớn là các đột biến Lành tính (Benign/Likely benign).

Trong bối cảnh này, nếu sử dụng phương pháp huấn luyện thông thường trên tập dữ liệu này sẽ dẫn đến hiện tượng mô hình bị lệch hoàn toàn về lớp đa số. Cụ thể, nếu mô hình tự động kết luận toàn bộ 100% mẫu kiểm tra đều là "Lành tính", nó vẫn đạt mức độ chính xác lên tới 88,2%. Điều này tạo ra kết quả đánh giá sai lệch và hoàn toàn không có giá trị trong chẩn đoán y khoa.

Để giải quyết rủi ro này, hệ thống áp dụng kỹ thuật cân bằng dữ liệu thông qua bộ lấy mẫu theo trọng số nghịch đảo tần suất (weighted_sampler). Phương pháp này can thiệp trực tiếp vào quá trình cung cấp dữ liệu cho mô hình: thay vì lấy dữ liệu ngẫu nhiên, hệ thống chủ động tăng tần suất xuất hiện của các mẫu Gây bệnh sao cho số lượng mẫu Gây bệnh và Lành tính mà mô hình phân tích trong mỗi chu kỳ là ngang bằng nhau.

Bên cạnh chiến lược dữ liệu, quá trình huấn luyện được duy trì tính ổn định thông qua các kỹ thuật tối ưu hóa hiện đại. Hệ thống sử dụng thuật toán tối ưu AdamW để cập nhật thông số mạng lưới, kết hợp cùng các kỹ thuật điều chỉnh tốc độ học (Cosine scheduler hoặc OneCycle) và kỹ thuật Warmup để giúp mô hình tiếp thu dữ liệu từ từ trước khi tăng tốc. Kỹ thuật Gradient clipping cũng được áp dụng nhằm khống chế mức độ thay đổi tối đa của các thông số toán học, giúp bảo vệ mạng 1D-CNN khỏi hiện tượng tính toán mất kiểm soát gây sụp đổ hệ thống.

### Lựa chọn hệ đo lường và Ngưỡng quyết định

Do sự sai lệch lớn về tỷ lệ lớp, thang đo Độ chính xác (Accuracy) ở ngưỡng mặc định (0.5) hoàn toàn bị loại bỏ khỏi tiêu chuẩn đánh giá chính. Dự án chuyển trọng tâm đánh giá sang các chỉ số phản ánh đúng năng lực nhận diện nhóm dữ liệu thiểu số (nhóm Gây bệnh):

- PR-AUC (Diện tích dưới đường cong Precision-Recall): Là chỉ số quan trọng hàng đầu trong các bài toán dữ liệu mất cân bằng. Nó đo lường khả năng mô hình duy trì tỷ lệ dự đoán đúng cao trong khi vẫn cố gắng tìm ra càng nhiều ca bệnh càng tốt.
- ROC-AUC (Diện tích dưới đường cong ROC): Đánh giá năng lực phân loại và tách biệt tổng quát giữa nhóm Gây bệnh và nhóm Lành tính của toàn hệ thống.
- Ngưỡng quyết định động (Decision Threshold): Thông thường, máy tính coi xác suất trên 50% là Gây bệnh và dưới 50% là Lành tính. Tuy nhiên, dự án không dùng mức 50% mặc định này. Hệ thống tự động chạy thử nghiệm trên tập dữ liệu Xác thực để tìm ra một mốc phần trăm tối ưu nhất giúp cân bằng giữa số lượng ca bệnh tìm được và mức độ chính xác của cảnh báo. Mốc phần trăm này sau đó mới được áp dụng để tính toán các chỉ số Precision, Recall và F1-Score.

### Kết quả thực nghiệm và Ma trận nhầm lẫn

Thử nghiệm thực tế của cấu trúc mô hình tối ưu (sử dụng dữ liệu đại diện từ cấu trúc benchmark tương đương với cửa sổ 601 bp, Dilated 1D-CNN, kỹ thuật học trên mạch DNA bổ sung ngược và phân tách dữ liệu nghiêm ngặt) đã cung cấp các kết quả hiệu năng cao.

Bảng 3.1: Các chỉ số hiệu suất trên tập Kiểm thử

| Chỉ số đánh giá               | Kết quả thực nghiệm |
| ----------------------------- | ------------------- |
| Test ROC-AUC                  | 0.9839              |
| ---                           | ---                 |
| Test PR-AUC                   | 0.9124              |
| ---                           | ---                 |
| Test F1-Score (Lớp Gây bệnh)  | 0.8178              |
| ---                           | ---                 |
| Test Precision (Độ chuẩn xác) | 0.7920              |
| ---                           | ---                 |
| Test Recall (Độ phủ)          | 0.8454              |
| ---                           | ---                 |

Để có cái nhìn sâu sắc hơn về hành vi dự đoán, Ma trận nhầm lẫn (Confusion Matrix) trên tập kiểm thử độc lập (tại ngưỡng Validation-F1) ghi nhận các phân bố cụ thể sau:

- Dự đoán đúng mẫu Lành tính (True Benign): 186.995 mẫu.
- Dự đoán đúng mẫu Gây bệnh (True Pathogenic): 20.308 mẫu.
- Cảnh báo sai (False Positive - Dương tính giả): 5.335 mẫu (Dự đoán nhầm Lành tính thành Gây bệnh).
- Bỏ sót bệnh (False Negative - Âm tính giả): 3.713 mẫu (Dự đoán nhầm Gây bệnh thành Lành tính).

Ý nghĩa và giá trị lâm sàng:

Chỉ số độ phủ đạt 84,54% cho thấy năng lực của mô hình là rất lớn khi nhận diện và cảnh báo thành công xấp xỉ 84,5% số ca mang đột biến gây bệnh thực sự tồn tại trong tập kiểm thử. Quan trọng hơn, Độ chuẩn xác (Precision) đạt 79,20% cho thấy hệ số tin cậy của cảnh báo rất cao. Về mặt toán học, điều này đồng nghĩa với việc: trong số 5 đột biến mà AI này đánh cờ cảnh báo là "Pathogenic", có tới 4 đột biến là hoàn toàn chính xác theo tiêu chuẩn của các chuyên gia.

Bên cạnh đó, nhóm 3.713 biến thể bị bỏ sót (Âm tính giả) cung cấp một nguồn dữ liệu quý giá cho hoạt động phân tích lỗi. Trong y sinh học, việc bỏ sót mầm bệnh mang rủi ro lớn hơn cảnh báo nhầm, do đó việc cô lập các trường hợp mô hình sai dự đoán sẽ là tiền đề để thiết lập các hệ số rủi ro an toàn và nâng cấp hệ thống trong các phiên bản tiếp theo.

### Khả năng diễn giải mô hình

Một rào cản lớn khi đưa các mô hình Học sâu vào lĩnh vực y tế là các bác sĩ không thể biết AI dựa vào cơ sở nào để đưa ra chẩn đoán. Để minh bạch hóa cơ chế dự đoán, dự án đã tích hợp phương pháp phân tích đạo hàm hướng dẫn (Guided Saliency Gradient) nhằm tạo ra các bản đồ nhiệt tính năng (Feature Heatmaps).

Phương pháp này tính toán đạo hàm của giá trị dự đoán đầu ra ngược về không gian ma trận 8-channel đầu vào, cho phép trực quan hóa mức độ đóng góp (importance) của từng vị trí nucleotide và từng kênh trình tự dọc theo cửa sổ 601 bp. Kết quả phân tích Saliency Gradient xác nhận rõ ràng rằng trọng số chú ý của mạng Dilated 1D-CNN hội tụ mạnh mẽ nhất xung quanh khu vực tọa độ trung tâm (nơi chứa tín hiệu của đột biến SNV) thay vì học vẹt các nhiễu nền ở vùng rìa xa của cửa sổ.

Sự tập trung phân bổ trọng số này chứng minh về mặt toán học rằng, mạng nơ-ron đã thực sự triết xuất được các motif đột biến sinh học cốt lõi từ trình tự nguyên bản, đáp ứng toàn diện tiêu chuẩn về khả năng diễn giải của một hệ thống chẩn đoán y khoa.

## Đánh giá độ tin cậy của chỉ số

Các chỉ số PR-AUC và ROC-AUC cao trên ClinVar không tự động chứng minh rằng mô hình đã học được bản chất sinh học của đột biến. Cơ sở dữ liệu ClinVar mang nhiều thiên lệch hệ thống: thiên lệch lấy mẫu (ascertainment bias — biến thể gây bệnh dồn vào số ít gen đã được nghiên cứu kỹ), hiện tượng vòng lặp (circularity) khi nhãn được gán một phần dựa trên chính các đặc trưng mà mô hình học (độ bảo tồn, tần số quần thể), và phần lớn mẫu âm là các trường hợp dễ phân biệt. Theo Grimm và cộng sự (2015), việc đánh giá các công cụ dự đoán biến thể bị cản trở bởi hai dạng vòng lặp: *Type-1 circularity* (cùng một biến thể xuất hiện ở cả tập huấn luyện và đánh giá) và *Type-2 circularity* (các biến thể khác nhau nhưng cùng một gen, mà gen đó gần như được gán đồng nhất một nhãn). Do đó, một chỉ số cao có thể phản ánh việc mô hình khớp với "bản đồ nhãn" hiện tại hơn là hiểu tác động của từng biến thể. Phần này trình bày bốn phép kiểm chứng độc lập nhằm định lượng phần nào của hiệu năng là đáng tin.

### Kiểm chứng theo thời gian

Sử dụng phép phân tách hai phiên bản ClinVar (train ≤ 2022-12, test = biến thể mới đến 2026-05) như đã mô tả ở mục chống rò rỉ dữ liệu, trên 586.778 biến thể mới mô hình đạt **ROC-AUC = 0,984** và **PR-AUC = 0,900** — gần như không suy giảm so với phép chia genome-block (ROC ≈ 0,984; PR-AUC ≈ 0,912). Kết quả này đồng thời chứng minh hai điều: (1) lượng dữ liệu huấn luyện ~610k là đủ (nếu thiếu, chỉ số đã sụt giảm rõ rệt), và (2) mô hình khái quát hóa tốt sang các biến thể được công bố trong tương lai, bác bỏ giả thuyết hiệu năng cao chủ yếu đến từ việc khớp ảnh chụp dữ liệu một thời điểm.

### Đối chứng âm (Negative Control)

Để xác minh mô hình thực sự sử dụng tín hiệu biến thể chứ không đạt điểm cao do lỗi rò rỉ quy trình, các thành phần tín hiệu được phá huỷ có chủ đích rồi đo mức suy giảm trên 50.000 mẫu kiểm thử:

| Phép đối chứng | ROC-AUC | PR-AUC |
| --- | --- | --- |
| Đầu vào đầy đủ | 0,984 | 0,901 |
| Tráo ALT tại tâm | 0,911 | 0,550 |
| Phá REF + ALT tại tâm | 0,861 | 0,421 |
| Xáo trộn nhãn (label shuffle) | 0,545 | 0,113 |

Phép xáo trộn nhãn đưa ROC-AUC về **0,545 ≈ 0,5** (mức ngẫu nhiên), khẳng định **không có rò rỉ trong quy trình huấn luyện**. Việc phá huỷ tín hiệu tại tâm làm hiệu năng suy giảm mạnh, chứng tỏ mô hình thực sự dựa vào thông tin biến thể.

### Phân rã đóng góp: ngữ cảnh vùng và biến thể cụ thể

Trường hợp phá huỷ cả REF và ALT tại tâm tương ứng với việc mô hình chỉ còn ngữ cảnh 600 bp xung quanh (biến thể bị xoá). Tách phần năng lực phân biệt vượt mức ngẫu nhiên (ROC − 0,5):

- Chỉ ngữ cảnh: 0,5 → 0,861, đóng góp +0,361 (**≈ 75%**).
- Thêm biến thể cụ thể: 0,861 → 0,984, đóng góp +0,123 (**≈ 25%**).

Như vậy khoảng **75%** năng lực phân biệt đến từ ngữ cảnh vùng và chỉ **25%** từ bản thân biến thể tại tâm. Mô hình thiên về vai trò bộ chấm điểm mức vùng (nhận biết "đây có phải vùng bảo tồn/nguy hiểm không") hơn là bộ dự đoán tác động của từng đột biến. Phần ngữ cảnh này không hoàn toàn vô nghĩa — vùng bảo tồn thực sự dự báo tính gây bệnh — nhưng nó là tín hiệu mức vùng. (Lưu ý: phép phá huỷ bằng cách nhồi tín hiệu sai có thể ước lượng hơi lệch; kết luận định tính "ngữ cảnh chiếm phần lớn" là vững chắc.)

### Tổng quát hóa sang gen mới (Seen/Unseen)

Để kiểm tra hiện tượng ghi nhớ theo gen (Type-2 circularity), tập kiểm thử được phân tầng thành nhóm gen đã xuất hiện trong tập huấn luyện (seen) và nhóm gen chưa từng xuất hiện (unseen). So sánh bằng ROC-AUC và lift (= PR-AUC / prevalence) do tỉ lệ dương khác nhau giữa hai nhóm.

| Nhóm | n | prevalence | PR-AUC | lift | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| gene_seen | 485.304 | 0,094 | 0,912 | 9,71 | 0,986 |
| gene_unseen | 101.474 | 0,074 | 0,841 | 11,35 | 0,972 |

Đọc theo ROC-AUC, khoảng cách seen–unseen chỉ **0,014**. Khác biệt PR-AUC (0,071) chủ yếu do prevalence thấp hơn ở nhóm unseen — minh chứng là lift của nhóm unseen (11,35) còn cao hơn nhóm seen (9,71). Do đó mức độ ghi nhớ theo gen là **nhỏ**: mô hình chuyển giao được sang các gen chưa từng thấy lúc huấn luyện.

### Tổng hợp

| Phép kiểm chứng | Loại thiên lệch kiểm soát | Kết quả | Kết luận |
| --- | --- | --- | --- |
| Temporal | Rò rỉ thời gian | ROC 0,984 không suy giảm | Khái quát theo thời gian tốt; data đủ |
| Negative control | Rò rỉ quy trình | Xáo nhãn → 0,545 | Không rò rỉ pipeline |
| Seen/Unseen | Ghi nhớ gen (Type-2) | Chênh ROC 0,014 | Ghi nhớ gen nhỏ |
| Phân rã 75/25 | Shortcut ngữ cảnh vùng | 75% từ ngữ cảnh | Còn tồn tại |

Ba phép đầu cho kết quả tích cực: phần lớn hiệu năng là thật, không do rò rỉ hay ghi nhớ gen. Phép thứ tư chỉ ra giới hạn còn lại — phần lớn năng lực phân biệt đến từ ngữ cảnh vùng (chủ yếu là tín hiệu độ bảo tồn), một dạng vòng lặp mà các phép chia theo không gian và thời gian không bóc tách được.

# Kết luận

Dự án đã hoàn thành mục tiêu xây dựng một hệ thống học sâu tự động phân loại tính gây bệnh của các đột biến gen đơn nucleotide (SNV) dựa trên dữ liệu lâm sàng ClinVar. Từ những thử nghiệm cơ sở ban đầu, dự án đã phát triển thành một quy trình phân loại chuỗi di truyền hoàn chỉnh và mang tính ứng dụng thực tiễn cao.

Về mặt xử lý dữ liệu và đánh giá mô hình: Hệ thống đã chuẩn hóa thành công quy trình tiền xử lý, đảm bảo mọi biến thể đều được đối chiếu chính xác tuyệt đối với hệ gen tham chiếu GRCh38. Nhằm giải quyết bài toán rò rỉ dữ liệu - một sai lầm phổ biến trong các nghiên cứu tin sinh học - dự án đã áp dụng chiến lược phân tách không gian theo khối gen (genome-block split) kết hợp với các màng lọc khoảng cách nghiêm ngặt. Bên cạnh đó, việc nhận diện và xử lý hiện tượng mất cân bằng lớp (11.8% Gây bệnh so với 88.2% Lành tính) đã giúp dự án định hình lại hệ đo lường: loại bỏ Độ chính xác (Accuracy) mang tính ảo giác, chuyển sang tối ưu hóa các chỉ số chính xác hơn như PR-AUC và thiết lập ngưỡng quyết định động từ tập xác thực.

Về mặt kiến trúc thuật toán: Quá trình nghiên cứu đã đi qua một chặng đường nâng cấp kiến trúc liên tục: từ việc sử dụng các kênh dày đặc (Dense 9-channel), mã hóa vị trí, đến việc thử nghiệm các cấu trúc nén, cổng kiểm soát (SE/Gating), và cơ chế Window Attention. Sự hội tụ của các thực nghiệm này đã dẫn đến thiết kế tối ưu cuối cùng: Sparse ALT 8-channel Dilated CNN 601 bp.

Cấu trúc này là sự kết hợp hoàn hảo giữa ba yếu tố kỹ thuật lõi:

- Thiết kế ma trận thưa 8 kênh giúp giữ lại sự đối chiếu trực tiếp giữa gen gốc và gen đột biến ngay tại tâm điểm, đồng thời loại bỏ hoàn toàn dữ liệu lặp thừa để tối ưu hóa tài nguyên tính toán.
- Mạng tích chập giãn nở (Dilated 1D-CNN) giúp mô hình có cái nhìn bao quát toàn bộ ngữ cảnh đoạn gen 601 bp mà không làm mất đi định vị không gian chính xác của điểm đột biến.
- Cơ chế học đối ngẫu mạch đôi (Reverse-Complement Augment và TTA) giúp mô hình hiểu được cấu trúc vật lý của phân tử DNA, đảm bảo tính ổn định tuyệt đối trong kết quả dự đoán.

Về hiệu năng và ý nghĩa lâm sàng: Trên tập dữ liệu kiểm thử độc lập, mô hình đạt chỉ số ROC-AUC xuất sắc ở mức 0.9839 và PR-AUC đạt 0.9124. Tại ngưỡng quyết định tối ưu, hệ thống ghi nhận Độ phủ (Recall) đạt 84.54% và Độ chuẩn xác (Precision) đạt 79.20%. Các kết quả này minh chứng rằng hệ thống không chỉ có năng lực nhận diện thành công phần lớn các mầm bệnh di truyền ẩn sâu trong chuỗi DNA, mà còn đảm bảo tỷ lệ cảnh báo sai ở mức rất thấp. Đây là tiền đề kỹ thuật để tích hợp mô hình này vào các công cụ hỗ trợ quyết định lâm sàng.

Hạn chế: Mặc dù được kiểm chứng kỹ về độ tin cậy, hệ thống còn các hạn chế cần nêu rõ:

- **Chưa loại bỏ được vòng lặp độ bảo tồn (conservation circularity).** Các phép chia theo khối gen và theo thời gian chỉ kiểm soát rò rỉ về không gian và thời gian. Phép phân rã đóng góp cho thấy khoảng 75% năng lực phân biệt đến từ ngữ cảnh vùng — chủ yếu là tín hiệu độ bảo tồn. Vì nhãn ClinVar cũng được gán một phần dựa trên độ bảo tồn, tồn tại một vòng lặp giữa đặc trưng mô hình học và tiêu chí gán nhãn. Vòng lặp này khái quát qua mọi gen nên không hiện ra ở phân tích seen/unseen, và chỉ có thể bóc tách bằng kiểm chứng chéo nguồn (cross-source) với dữ liệu độc lập như tần số quần thể gnomAD hoặc dữ liệu thực nghiệm.
- **Mô hình thiên về mức vùng hơn mức biến thể.** Bản thân biến thể cụ thể chỉ đóng góp khoảng 25% năng lực phân biệt; mô hình mạnh ở việc nhận biết "vùng nguy hiểm" nhưng còn hạn chế trong phân giải tác động của từng đột biến đơn lẻ trong cùng một vùng.
- **Chỉ dùng thông tin chuỗi (sequence-only).** Hệ thống chưa tích hợp các chú thích sinh học quan trọng như độ bảo tồn (phyloP/phastCons), hệ quả chức năng (VEP consequence), điểm splicing (SpliceAI) hay tần số quần thể (gnomAD). Đây là trần hiệu năng của hướng tiếp cận thuần chuỗi.
- **Phụ thuộc chất lượng nhãn ClinVar.** Dù đã lọc theo review status, một phần các trường hợp dự đoán sai với độ tự tin cao có thể bắt nguồn từ nhiễu nhãn (một submitter, nhãn "Likely", đánh giá cũ) hơn là lỗi thực sự của mô hình.
- **Bài toán đã được đơn giản hóa.** Nghiên cứu chỉ xét SNV và phân loại nhị phân, đã loại bỏ các nhãn không chắc chắn (Uncertain significance, Conflicting). Đây là nghiên cứu mang tính phương pháp/học thuật, chưa phải hệ thống đạt chuẩn lâm sàng.

Định hướng phát triển tương lai: Mặc dù đã đạt được hiệu năng cao, dự án vẫn còn những điểm để tiếp tục hoàn thiện trong các nghiên cứu tiếp theo:

- Đánh giá hiệu chuẩn mô hình: Cần tiến hành đo lường các chỉ số như Brier score và Expected Calibration Error để đảm bảo xác suất do mô hình xuất ra thực sự phản ánh đúng tỷ lệ rủi ro trong thực tế, giúp bác sĩ tin tưởng hơn khi tham khảo.
- Phân tích lỗi chuyên sâu: Tập trung mổ xẻ nhóm 3.713 ca âm tính giả (những đột biến gây bệnh nhưng hệ thống dự đoán nhầm là lành tính). Việc phân tích cụ thể các mẫu mà mô hình dự đoán sai sẽ giúp tìm ra giới hạn của mô hình để tinh chỉnh.
- Tích hợp khả năng diễn giải trực tiếp: Đóng gói thuật toán tính toán Bản đồ nhiệt vào cùng một luồng suy diễn của mô hình, cho phép xuất báo cáo trực quan cho từng bệnh nhân.

# Tài liệu tham khảo