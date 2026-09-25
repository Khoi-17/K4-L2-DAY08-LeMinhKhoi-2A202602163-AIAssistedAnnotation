# Vì sao chọn lô này?

Trong 50 dòng đứng đầu `outputs/selection_round1.csv`, chọn năm frame bạn sẽ ưu tiên nếu chỉ có
ngân sách rà năm ảnh. Ghi tên, điểm, thời điểm, thứ tự và lý do; tối thiểu một quyết định phải xét
ảnh gần trùng hoặc trường hợp model không dự đoán được box:
1. `frame_0182.jpg` (rank 1, score 0.9591, t = 72.8s): Điểm tổng hợp cao nhất toàn bộ pool, độ bất định U = 0.9182, có tới 18 box mơ hồ (ambiguous) và mật độ 28 box. Đây là frame nhiều xe tối và ánh đèn phức tạp cần người rà soát nhất.
2. `frame_0369.jpg` (rank 2, score 0.9324, t = 147.6s): Độ bất định cao (U = 0.9315), mật độ box lớn (43 box, 16 box mơ hồ), đại diện cho phân đoạn giao thông đông đúc về cuối video.
3. `frame_0380.jpg` (rank 3, score 0.9170, t = 152.0s): Cách frame 0369 là 4.4s (đảm bảo vượt ngưỡng `MIN_GAP_S`), có 40 box với 15 box mơ hồ, phản ánh tình trạng đèn xe chói lóa.
4. `frame_0326.jpg` (rank 4, score 0.9155, t = 130.4s): Đại diện cho phân khúc giữa video (t = 130.4s), chứa 39 box với 15 box mơ hồ.
5. `frame_0099.jpg` (rank 8, score 0.9063, t = 39.6s): Mặc dù rank 8 thấp hơn frame 0372 (rank 6) và frame 0312 (rank 7), nhưng frame 0372 bị gần trùng với frame 0369, còn frame 0099 nằm ở mốc t = 39.6s (đoạn đầu video). Việc ưu tiên frame 0099 giúp phân bổ ngân sách 5 ảnh trải đều trên toàn bộ trục thời gian thay vì dồn hết vào nửa sau video (t > 120s).

Ba frame thuộc lô 12 ảnh model chọn và bằng chứng trong CSV/ảnh contact sheet:
- `frame_0182.jpg` (rank 1, t = 72.8s, score = 0.9591): Đứng đầu CSV với U = 0.9182, A = 1.0, D = 1.0. Trong contact sheet, frame này có lượng lớn xe tối chạy ở làn phải và nhiều vệt sáng đèn hậu bị nhiễu.
- `frame_0331.jpg` (rank 5, t = 132.4s, score = 0.9154): Có tới 47 box và 18 box mơ hồ (A = 1.0). Trong contact sheet, mặt đường phản chiếu ánh đèn mạnh khiến AI dao động giữa vật thể thật và vệt sáng phản quang.
- `frame_0107.jpg` (rank 14, t = 42.8s, score = 0.8876): Nằm ở đầu video, cách frame 0099 đúng 3.2s (> MIN_GAP_S = 2.0s). Dữ liệu CSV cho thấy có 33 box và 15 box mơ hồ; contact sheet cho thấy nhiều xe ở xa mờ chỉ thấy đèn pha.

Một frame có điểm cao nhưng không chọn hoặc một frame có điểm thấp vẫn nên xem, và lý do:
- `frame_0368.jpg` (rank 9, score = 0.9003, t = 147.2s) và `frame_0372.jpg` (rank 6, score = 0.9101, t = 148.8s): Cả hai frame này có điểm số rất cao (> 0.90) nhưng đều không được chọn (`selected = False`) vì nằm quá sát `frame_0369.jpg` (t = 147.6s, độ lệch thời gian chỉ 0.4s và 1.2s, vi phạm ngưỡng giãn cách `MIN_GAP_S = 2.0s`). Đây là các ảnh gần trùng (near-duplicates); việc bỏ qua chúng giúp tiết kiệm công rà nhãn cho những frame mang thông tin mới.

Điều phép chọn này chưa chứng minh về chất lượng mô hình:
- Điểm chọn mẫu (score) phản ánh độ bất định của mô hình và tính đa dạng hình học/thời gian, nhưng không đảm bảo rằng fine-tune trên các mẫu này chắc chắn làm tăng số đo (như AP50) trên tập kiểm thử. Độ bất định có thể xuất phát từ nhiễu dữ liệu (nhiễu phản quang, xe quá xa bị mờ nhoè) chứ không hẳn là tri thức hữu ích. Nếu mô hình học theo các trường hợp quá dị biệt hoặc phân phối tập test không đồng nhất với các ca khó này, độ chính xác có thể giảm cục bộ.
