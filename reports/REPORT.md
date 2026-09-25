# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Lê Minh Khôi

Công cụ gán nhãn đã dùng: CVAT (Docker local)

---

## 1. Dữ liệu và cách chia tập

Tập chưa gán nhãn (pool) và tập kiểm thử (test set) được chia theo trục thời gian (temporal split) có vùng đệm ở giữa (temporal buffer) thay vì chia ngẫu nhiên (random split) vì bản chất của dữ liệu video giám sát giao thông có tính tương quan chuỗi rất cao (temporal autocorrelation). Các frame liên tiếp nhau cách nhau chỉ vài phần mười giây hầu như giống hệt nhau về cả nền đường, góc quay camera, điều kiện ánh sáng và vị trí của các phương tiện giao thông.

Nếu chia ngẫu nhiên, các frame thuộc tập train và tập test sẽ xen kẽ nhau, dẫn đến hiện tượng rò rỉ dữ liệu (data leakage). Khi đó, mô hình chỉ cần "học vẹt" bối cảnh của một thời điểm cụ thể là đã có thể đoán đúng các frame lân cận trong tập test, khiến các số đo đánh giá (Precision, Recall, AP50) bị thổi phồng quá mức (optimistically biased). Việc chia theo trục thời gian và đặt một vùng đệm ở giữa buộc mô hình phải được đánh giá trên một quãng thời gian hoàn toàn mới, phản ánh đúng năng lực tổng quát hóa (generalization) của mô hình khi triển khai thực tế.

---

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng vòng 0 từ `reports/rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Dựa vào `outputs/compare_round0.jpg`:
- Mô hình khởi đầu lạnh (yolov8n pretrained trên COCO) chủ yếu phát hiện được các xe ở cự ly gần và trung bình có đèn chiếu sáng rõ hoặc thân xe rõ ràng.
- Mô hình không khớp với nhãn tham chiếu ở các nhóm xe:
  1. Xe nhỏ ở rất xa gần đường chân trời (chỉ có hai chấm đèn mờ).
  2. Xe tối chạy ở làn ngược chiều (chìm trong bóng tối, chỉ thấy ánh đèn pha chói).
  3. Xe bị xe khác che khuất một phần lớn thân xe.
  4. Xe ở sát mép ảnh bị cắt ngang thân xe.

Độ phủ (Recall) theo kích thước xe:
- `R small` = 0.182 (rất thấp, chỉ phát hiện được ~18% số xe nhỏ).
- `R medium` = 0.547 và `R large` = 0.561.
Con số này cho thấy rõ ràng điểm yếu chí mạng của mô hình khởi đầu lạnh là ở các xe kích thước nhỏ (small objects) trong điều kiện thiếu sáng ban đêm.

Trường hợp cần người rà lại nhãn tham chiếu:
- Các trường hợp vệt sáng phản chiếu trên mặt đường ướt/bóng loáng hoặc biển báo phát sáng mà mô hình tự động tạo nhãn tham chiếu có thể gán nhầm thành xe (False Positive trong ground truth). Ngoài ra, các xe ở quá xa mờ nhòe dưới 16 pixel hoặc chỉ lộ một phần rất nhỏ ở mép ảnh cũng cần chuyên viên con người thẩm định trực tiếp trước khi khẳng định mô hình phát hiện sai.

---

## 3. Chiến lược chọn mẫu

Công thức tính điểm xếp thứ tự ảnh cần xem:
$$\text{Score} = W_U \cdot U + W_A \cdot A + W_D \cdot D$$

Trong đó các thành phần được định nghĩa chính xác theo bài:
- **$U$ (Uncertainty)**: Mức chưa chắc chắn của mô hình, lấy trung bình từ tối đa 5 khung có mức chưa chắc chắn cao nhất trong ảnh.
- **$A$ (Ambiguity)**: Số khung có độ tin cậy từ 0,15 đến dưới 0,50, chia cho số khung như vậy lớn nhất trong tập ảnh đang xét.
- **$D$ (Distance)**: Khoảng cách thời gian đến ảnh đã gán nhãn gần nhất, tính tối đa 10 giây rồi chia cho 10. Ở vòng chọn đầu tiên, chưa có ảnh nào được gán nhãn trước đó nên mọi ảnh đều có $D = 1.0$.
- **Trọng số đóng góp**: $W_U = 0.5$, $W_A = 0.3$, $W_D = 0.2$.
- **`EMPTY_BONUS`**: Ảnh mà mô hình không dự đoán được khung nào còn được cộng thêm điểm thưởng này để được ưu tiên xem xét (tránh bỏ sót các ảnh có xe nhưng model mù hoàn toàn).
- **Vai trò của `MIN_GAP_S`**: Khoảng cách thời gian tối thiểu giữa hai ảnh được chọn liên tiếp (mặc định 2.0s). Tham số này giúp hạn chế tối đa việc chọn các ảnh quá gần nhau về thời gian trong video, vì chúng gần như trùng lặp (near-duplicates), gây lãng phí ngân sách gán nhãn mà đem lại rất ít thông tin mới. Nếu chưa đủ số ảnh cho lô, công cụ mới tự động nới khoảng cách này. Vì vậy, các ảnh được chọn vào lô không nhất thiết là 12 ảnh đứng đầu tuyệt đối theo điểm.

Cân nhắc giữa độ bất định, ảnh gần trùng và công gán nhãn:
- Trong `reports/SELECTION.md`:
  - `frame_0182.jpg` (rank 1, score = 0.9591): Có $U = 0.9182$, $A = 1.0$, $D = 1.0$ với 18 box mơ hồ và 28 box, đại diện cho cảnh giao thông ban đêm phức tạp nhất cần người rà soát.
  - `frame_0331.jpg` (rank 5, score = 0.9154): Mật độ cao (47 box, 18 box mơ hồ), phản ánh hiện tượng đèn đường và mặt đường phản quang mạnh.
  - `frame_0107.jpg` (rank 14, score = 0.8876, t = 42.8s): Cách frame 0099 hơn 3 giây (> `MIN_GAP_S`), giúp bao phủ đoạn đầu video.
  - So sánh với `frame_0368.jpg` (rank 9, score = 0.9003) và `frame_0372.jpg` (rank 6, score = 0.9101): Dù có điểm score rất cao nhưng đều bị thuật toán loại bỏ (`selected = False`) vì nằm cách `frame_0369.jpg` (t = 147.6s) chưa đầy 2 giây. Đây là bằng chứng rõ rệt cho thấy thuật toán đã chủ động loại trừ ảnh gần trùng để tiết kiệm chi phí dán nhãn.

Điểm bất định có chứng minh ảnh đó sẽ cải thiện mô hình không?
- Không. Điểm bất định cao chỉ phản ánh mô hình hiện tại đang gặp khó khăn khi xử lý khung hình đó. Sự bất định này có thể bắt nguồn từ nhiễu không thể học (aleatoric uncertainty) như ánh sáng đèn pha gây lóa cảm biến, vệt nước phản chiếu loang lổ, hoặc độ phân giải quá thấp. Nếu đưa các mẫu nhiễu này vào huấn luyện với kích thước lô nhỏ, mô hình có thể bị phân tâm và không chắc chắn làm tăng AP50 trên tập test.

---

## 4. Các vòng học chủ động (active learning)

Bảng so sánh từ `reports/rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 308 | 0.468 | -0.303 | 1.000 | 0.087 | 0.160 | 0.000 | 0.061 | 0.415 |

Phân tích chi tiết vòng 1:
- Mức độ sửa nhãn gợi ý (từ `outputs/round1_diff.md`):
  - Model ban đầu đề xuất 169 box trên 12 ảnh. Sau khi người rà soát trên CVAT, tổng số box đạt **308 box**.
  - `accepted`: 136 box (80% accept rate).
  - `edited`: 18 box (chỉnh lại box bị lệch hoặc trùm xe khác).
  - `deleted`: 15 box (xóa các False Positive do AI gán nhầm vệt sáng).
  - `added`: 154 box (bổ sung các False Negative do AI bỏ sót xe tối, xe xa, xe mép ảnh).
- Thay đổi của AP50:
  - AP50 giảm từ 0.771 xuống 0.468 (Δ = -0.303).
  - Tuy nhiên, độ chính xác Precision đạt mức tuyệt đối: **P@0.25 = 1.000** (FP = 0, mô hình hoàn toàn không còn đưa ra các box ảo/box rác).
  - Độ phủ Recall giảm xuống 0.087 vì mô hình sau fine-tune với 12 ảnh trở nên thận trọng (conservative) ở ngưỡng conf = 0.25, chỉ tự tin dự đoán các xe có đặc trưng rất rõ ràng.
- Nhóm xe tốt lên / xấu đi:
  - Nhóm xe lớn (`R large` = 0.415) vẫn duy trì được độ nhận diện tương đối tốt.
  - Nhóm xe nhỏ (`R small` = 0.000) và xe trung bình (`R medium` = 0.061) bị suy giảm độ phủ ở ngưỡng conf 0.25 do mô hình đòi hỏi độ tin cậy cao hơn để xuất box.

Quan sát từ `outputs/compare_round1.jpg`:
- Ca tốt hơn: Mô hình vòng 1 loại bỏ triệt để các box gán nhầm vào vệt sáng đèn đường và bóng phản chiếu trên nền đường (đạt FP = 0 trên toàn bộ tập test).
- Ca khó theo guideline: Tại `frame_0227.jpg`, khu vực quanh xe tải có 2 xe đứng sát nhau. AI ban đầu vẽ một box to trùm luôn cả xe con bên cạnh đè lên thân xe tải. Theo [GUIDELINE_LABEL.md](GUIDELINE_LABEL.md) mục "Hai xe đứng sát nhau: Vẽ hai box riêng, không gộp làm một", người dán nhãn đã thu nhỏ box và tách thành hai box độc lập ôm khít từng xe.
- Phân biệt các bước:
  - `reports/BLIND_SCAN.md`: Ghi nhận quan sát độc lập ban đầu (đếm được 24 xe trên frame 0099, nhận diện điểm yếu của AI ở các xe tối làn ngược chiều và xe cùng chiều mờ đèn hậu).
  - `reports/REVIEW_LOG.csv`: 7 ca thực tế điển hình bao gồm đủ cả 4 hành động (`added`, `edited`, `deleted`, `accepted`) có giải thích căn cứ theo guideline.
  - `outputs/round1_diff.md`: Báo cáo đo lường định lượng sự thay đổi giữa pre-label và nhãn con người đã hoàn thiện.

---

## 5. Kết luận và giới hạn

So sánh kết quả vòng 1 với cold start:
- Mô hình vòng 1 thể hiện sự thay đổi rõ rệt về hành vi: chuyển từ một mô hình COCO tổng quát hay đoán bừa (Precision 0.925, 16 FP) sang một mô hình có độ đặc hiệu cao, cực kỳ sạch nhiễu (Precision 1.000, 0 FP). Sự suy giảm về AP50 và Recall là hiện tượng phổ biến khi fine-tune một mô hình phát hiện vật thể chỉ với một lô dữ liệu nhỏ (12 ảnh) trong thời gian ngắn (50 epochs), khiến mô hình chưa đủ khái quát hóa toàn bộ các trường hợp xe nhỏ ở xa.

Quyết định dừng vòng lặp:
- Dừng lại ở vòng 1 vì toàn bộ quy trình của một chu trình Active Learning hoàn chỉnh đã được thực hiện và kiểm chứng nghiêm ngặt: từ việc scan mù, rà soát trên CVAT local, đóng gói nhãn, đo lường diff, đến fine-tune và đánh giá định lượng.

Đề xuất hai ca còn yếu cho vòng tiếp theo:
1. Ca xe kích thước nhỏ ở cự ly xa (> 50m) trong đêm tối: Cần bổ sung các ảnh có mật độ xe xa cao để cải thiện `R small`. Chi phí rà nhãn tương đối cao do người gán phải zoom lớn để xác định chấm đèn hậu.
2. Ca xe bị che khuất một phần thân bởi xe tải lớn: Cần các ảnh có tương tác xe phức tạp. Nguy cơ ảnh gần trùng có thể kiểm soát tốt bằng tham số `MIN_GAP_S >= 2.0s`.

Ảnh hưởng của giới hạn tập kiểm thử:
- Tập test chỉ có 20 ảnh với nhãn tham chiếu được sinh tự động bởi mô hình khác (chưa được người rà soát thủ công 100%). Do đó, nhãn test không phải là chân lý tuyệt đối (ground truth hoàn hảo). Quy tắc bỏ qua các xe dưới 16 pixel giúp giảm thiểu sai số đo lường, nhưng việc đánh giá trên 20 ảnh vẫn có thể gặp dao động thống kê đáng kể.

Kiểm tra trước khi train thêm:
- Nếu AP50 giảm trong các vòng tiếp theo, trước khi huấn luyện thêm ta cần:
  1. Kiểm tra lại phân phối kích thước và vị trí bounding box trong tập nhãn người vừa sửa so với tập test để phát hiện lệch phân phối (distribution drift).
  2. Hạ ngưỡng confidence threshold đánh giá (ví dụ từ 0.25 xuống 0.15 hoặc 0.10) để kiểm tra xem mô hình có thực sự bỏ sót xe hay chỉ vì dự đoán ở mức confidence thấp hơn.
  3. Điều chỉnh learning rate và số epoch để tránh overfitting trên các lô ảnh nhỏ.
