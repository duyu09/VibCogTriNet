# VibCogTriNet

[**Tiếng Trung Phồn Thể(繁體中文)**](./README.md) | [**Tiếng Anh(English)**](./README.en.md) | [**Tiếng Việt**](./README.vi.md)

**VibCogTriNet** là một mô hình học sâu có thể nhận dạng sóng rung của ổ bi để phát hiện lỗi ổ bi. Mô hình dựa trên sự kết hợp ba loại đặc trưng: miền thời gian, miền tần số và đặc trưng thống kê của sóng rung ổ bi, từ đó xác định loại lỗi ổ bi. Mô hình và phương án thiết kế này đã được sử dụng để tham gia Cuộc thi mô hình hóa toán học dành cho nghiên cứu sinh Cúp Hoa Vi(Tiếng Trung: 華為杯; Tiếng Anh: Huawei Cup) năm 2025, toàn bộ tập dữ liệu huấn luyện mô hình được cung cấp bởi ban tổ chức cuộc thi.

---

### **Sơ đồ quy trình xử lý dữ liệu:**

<img src="./images/data_processing.svg" style="width: 70%;">

---

### **Sơ đồ kiến trúc mô hình:**

<img src="./images/model_architecture.svg" style="width: 92%;">

---

### **Đường cong suy giảm hàm mất mát trong huấn luyện (42 Epoch đầu là học so sánh, 344 Epoch sau là học có giám sát)**

<img src="./images/training_loss_curve.svg" style="width: 70%;">

---

### **Biểu đồ hộp chuẩn hóa chuẩn độ dốc lan truyền xuôi của mô hình**

Bằng phương pháp quy kết gradient, ta tính toán gradient tại tầng kết nối đầy đủ sau khi ghép ba loại đặc trưng để đưa vào bộ phân loại. Từ đó có thể suy ra mức độ “đóng góp” của ba loại đặc trưng này đối với kết quả phân loại cuối cùng. Hình dưới đây cho thấy trên tập kiểm tra, phân bố gradient của đặc trưng miền thời gian, miền tần số và thống kê tại lớp mạng nơ-ron đầu tiên của bộ phân loại kết nối đầy đủ. Cả ba loại đặc trưng đều có gradient lớn hơn 0, chứng tỏ chúng đều tác động tích cực đến kết quả phân loại cuối cùng của mô hình.

<img src="./images/gradient_attribution_distribution.svg" style="width: 70%;">

---

### **Thí nghiệm Loại bỏ**

| Mô hình                        | Accuracy     | F1-score     |
| ------------------------------ | ------------ | ------------ |
| _**VibcogTriNet**_             | _**0.9969**_ | _**0.9538**_ |
| VibcogTriNet-no-Transformer    | 0.9812       | 0.9501       |
| VibcogTriNet-no-SpectrogramCNN | 0.9289       | 0.9001       |
| VibcogTriNet-no-Stats          | 0.9794       | 0.9444       |

---

### **Tác giả Dự án**

**BẢN QUYỀN &copy; 2025** [**HÀ Phi Phàm**](https://faculty.lzjtu.edu.cn/chenmei/zh_CN/xsxx/2554/content/1835.htm) (Tiếng Trung: 何非凡; Tiếng Anh: HE Feifan), [**ĐỖ Vũ**](https://faculty.lzjtu.edu.cn/chenmei/zh_CN/xsxx/2554/content/1837.htm) (Tiếng Trung: 杜宇; Tiếng Anh: DU Yu), [**DƯƠNG Sa Sa**](https://faculty.lzjtu.edu.cn/chenmei/zh_CN/xsxx/2554/content/1836.htm) (Tiếng Trung: 楊莎莎; Tiếng Anh: YANG Shasha), [**Đại Học Giao thông Lan Châu, Học Viện Điện Tử Và Công Nghệ Thông Tin**](https://dxxy.lzjtu.edu.cn/) (Tiếng Trung: 蘭州交通大學電子與信息工程學院; Tiếng Anh: School of Electronic and Information Engineering, Lanzhou Jiaotong University)
