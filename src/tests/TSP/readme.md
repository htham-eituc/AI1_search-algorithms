BẢNG PHÂN LOẠI 15 SUBTASK CHO TSP (1000 Tests)

## Subtask 1 (Tests 1 - 30): Đồ thị siêu nhỏ ($N \le 10$)

Đặc điểm: Đồ thị đầy đủ, trọng số ngẫu nhiên.

Mục đích: Dành cho thuật toán vét cạn hoán vị $O(N!)$ để sinh đáp án chuẩn (trâu) đối chứng.

## Subtask 2 (Tests 31 - 80): Giới hạn của DP Bitmask ($N \le 20$)

Đặc điểm: Đồ thị đầy đủ, $N$ từ 15 đến 20.

Mục đích: Test thuật toán Quy hoạch động $O(N^2 2^N)$. Nếu code DP không cẩn thận sẽ bị Time Limit Exceeded (TLE) hoặc cấp phát thiếu mảng nhớ.

## Subtask 3 (Tests 81 - 130): Đồ thị vòng (Cycle Graph) ($N \le 1000$)

Đặc điểm: Đồ thị chỉ có đúng $N$ cạnh tạo thành 1 chu trình duy nhất.

Mục đích: Chỉ tồn tại duy nhất 1 đáp án. Test xem thuật toán có tìm được chính xác con đường duy nhất này hay không.

## Subtask 4 (Tests 131 - 190): Đồ thị Ơ-clit (Euclidean TSP) ($N \le 300$)

Đặc điểm: Đồ thị đầy đủ. Trọng số là khoảng cách vật lý giữa 2 điểm tọa độ $(x, y)$ trên mặt phẳng.

Mục đích: Đồ thị thoả mãn Bất đẳng thức tam giác ($w(u,v) + w(v,w) \ge w(u,w)$). Các thuật toán xấp xỉ (như Christofides) sẽ hoạt động rất hiệu quả ở test này.

## Subtask 5 (Tests 191 - 250): Gần đầy đủ (Almost Complete) ($N \le 300$)

Đặc điểm: Đồ thị bị khuyết đi một vài cạnh ngẫu nhiên.

Mục đích: Bẫy các code luôn mặc định ma trận là đồ thị đầy đủ. Thuật toán cần phải xử lý trường hợp một số cạnh không tồn tại nhưng vẫn có thể tìm ra chu trình.

## Subtask 6 (Tests 251 - 310): Đồ thị hình sao (Star Graph) ($N \le 1000$)

Đặc điểm: Đỉnh trung tâm nối với tất cả, các đỉnh rìa không nối với nhau.

Mục đích: Chắc chắn không có chu trình Hamilton. Cần in ra kết quả -1 hoặc No solution.

## Subtask 7 (Tests 311 - 370): Đồ thị hai phía (Bipartite Complete) ($N \le 400$)

Đặc điểm: Chia làm 2 tập $X$ và $Y$ ($|X| = |Y|$). Chỉ có cạnh nối giữa $X$ và $Y$.

Mục đích: Chỉ đồ thị hai phía có hai tập đỉnh bằng nhau mới có chu trình Hamilton. Code Heuristic dễ bị "đi vào ngõ cụt" ở loại đồ thị này.

## Subtask 8 (Tests 371 - 430): Cạnh thưa ngẫu nhiên (Sparse Random) ($N \le 1000$)

Đặc điểm: Số lượng cạnh rất ít ($M \approx 3N$).

Mục đích: Đa số là không có chu trình Hamilton, test tốc độ thoát sớm (early exit) của thuật toán kiểm tra.

## Subtask 9 (Tests 431 - 490): Đồ thị vô hướng, Trọng số đồng nhất ($N \le 500$)

Đặc điểm: Mọi cạnh đều có chung trọng số là $1$.

Mục đích: Biến TSP thành bài toán tìm Chu trình Hamilton cơ bản. Không có sự khác biệt về trọng số, bẫy các thuật toán Tham lam (Greedy).

## Subtask 10 (Tests 491 - 560): Bẫy hai trọng số (Two-weight Trap) ($N \le 500$)

Đặc điểm: Trọng số chỉ nhận giá trị $1$ hoặc $10^6$.

Mục đích: Một đường đi tối ưu sẽ cố gắng chỉ chọn cạnh trọng số $1$. Thuật toán không khéo sẽ chọn nhầm cạnh $10^6$ và làm kết quả sai lệch khủng khiếp.

## Subtask 11 (Tests 561 - 630): TSP Bất đối xứng (Asymmetric TSP) ($N \le 300$)

Đặc điểm: Đồ thị CÓ HƯỚNG (Directed Graph), cạnh $u \to v$ có trọng số khác cạnh $v \to u$.

Mục đích: Rất nhiều bài TSP thực tế là một chiều (như đường một chiều, hoặc hướng gió bay).

## Subtask 12 (Tests 631 - 700): Đồ thị tràn số (Overflow Danger) ($N \le 500$)

Đặc điểm: Trọng số mỗi cạnh lên tới $10^{13}$.

Mục đích: Kiểm tra lỗi tràn số nguyên (Integer Overflow). Cần dùng long long để tính tổng chi phí.

## Subtask 13 (Tests 701 - 800): Đồ thị chia cụm (Clustered / Islands) ($N \le 500$)

Đặc điểm: Gồm nhiều cụm đỉnh nén chặt với nhau (trọng số nội bộ cực nhỏ), nối với cụm khác bằng một vài cạnh có trọng số rất lớn.

Mục đích: Bài toán "Cái bẫy hoàn hảo" cho thuật toán Tham lam chọn cạnh ngắn nhất (Nearest Neighbor) – thường sẽ bị mắc kẹt ở một cụm rồi buộc phải nhảy một bước chi phí khổng lồ ở chặng cuối.

## Subtask 14 (Tests 801 - 900): Max Size - Thưa ($N = 1000$, $M = 5000$)

Đặc điểm: Đồ thị rất lớn nhưng số lượng cạnh ít.

Mục đích: Stress test giới hạn bộ nhớ (Memory Limit) và Time Limit trên $O(N+M)$.

## Subtask 15 (Tests 901 - 1000): Max Size - Dày đặc ($N = 1000$, $M = \frac{N(N-1)}{2}$)

Đặc điểm: Khoảng 500,000 cạnh. Đồ thị vô hướng đầy đủ kích thước tối đa.

Mục đích: Stress test tốc độ đọc I/O (cin/cout) và thuật toán Heuristic lớn nhất có thể.