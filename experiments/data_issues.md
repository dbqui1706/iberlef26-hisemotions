# Data Issues — HISEMOTIONS 2026 (UPDATED: 2026-03-19)

> Phân tích chi tiết các đặc trưng dữ liệu ảnh hưởng đến quá trình huấn luyện mô hình.
> Tập dữ liệu đã đi qua bước Tiền xử lý (Preprocessing): Loại bỏ NaN, xử lý chuỗi ngắn, hợp nhất nhãn trùng lặp và THÊM nhãn `neutral`.
> Lớp nhãn mới: `['anger', 'fear', 'joy', 'sadness', 'surprise', 'hope', 'neutral']`
> Kích thước Train: 2,484 mẫu | Dev: 400 mẫu.

---

## 1. Dữ liệu đã sạch 100% (Clean Data)
Báo cáo tiền xử lý:
- Dữ liệu bị nhiễu NaN đã bị xóa bỏ, ngăn ngừa NaN Loss.
- Không còn câu văn bản ngắn < 3 ký tự.
- Samples bị trùng lặp Text đã được hợp nhất nhãn (bằng phép Toán OR).
- Tổng Row Train giảm từ 2659 -> 2484. Dev giảm 425 -> 400.

---

## 2. Phân bố nhãn đã thay đổi: Neutral là Lớp Đa Số
Trước đây, vì không có nhãn tường minh cho các câu không mang cảm xúc, model bị bắt "không dự đoán gì cả". Hiện tại, nhãn `neutral` đã trở thành lớp đứng đầu (1148 mẫu), định hình lại nhận thức của mô hình về phân bổ thực sự.

| Emotion | Train Count | % of Train | Dev Count | % of Dev | Severity |
|---------|------------:|-----------:|----------:|---------:|:---------|
| 🖤 **neutral** | 1,148 | 46.2% | 181 | 45.3% | ✅ OK |
| 😢 **sadness** | 1,139 | 45.9% | 70 | 17.5% | ⚠️ Train > Dev |
| 😨 **fear** | 293 | 11.8% | 40 | 10.0% | ✅ OK |
| 😊 **joy** | 121 | 4.9% | 29 | 7.3% | ✅ OK |
| 🌟 **hope** | 73 | 2.9% | 85 | 21.3% | 🔴 Dev >>> Train |
| 😡 **anger** | 11 | 0.4% | 52 | 13.0% | 🔴 Dev >>> Train |
| 😲 **surprise**| 7 | 0.3% | 7 | 1.8% | 🔴 Critical Rarest |

![Train vs Dev Distribution](../docs/images/01_train_vs_dev_distribution.png)
![Class Imbalance Ratio](../docs/images/03_class_imbalance_ratio.png)

> **Nhận xét:** Train cực kỳ mất cân bằng. Nhóm rare classes (anger, surprise, hope) quá ít mẫu so với Dev và với chính Train.

---

## 3. Train-Dev Distribution Mismatch Khủng hoảng

Sự chênh lệch giữa Train (LLM-annotated) và Dev (Human Gold standard) vẫn là một vấn đề cực kỳ căng thẳng chưa thể thay đổi trừ khi dùng Data Augmentation.

![Distribution Mismatch](../docs/images/02_distribution_mismatch.png)

- **Anger:** Dev chiếm 13%, nhưng Train chỉ có 0.4% (Thiếu hụt ~32.5 lần!).
- **Hope:** Dev chiếm 21.3%, Train chỉ 2.9% (Thiếu hụt ~7 lần!).
- **Sadness:** Train chiếm đến 45.9% nhưng tập Dev chỉ có 17.5%.

> **Tác động (Impact):** Mô hình có xu hướng thiên vị (Bias) dự đoán `neutral` và `sadness`, đồng thời học rất yếu `anger` và `hope`.

---

## 4. Đặc điểm Multi-Label (Số Nhãn trên Câu)

Vì chúng ta đã điền 1 vào `neutral` cho những câu [0,0,0,0,0,0], bây giờ *mỗi câu trong dataset đều có ít nhất 1 nhãn*.

![Labels per Sample](../docs/images/04_labels_per_sample.png)

- Phần lớn các câu chỉ chứa 1 cảm xúc duy nhất (`sadness`, `fear`, hoặc `neutral`).
- Cực kì ít câu có từ 2 cảm xúc trở lên.

![Label Co-occurrence](../docs/images/05_label_cooccurrence.png)

- Khi có 2 cảm xúc, sự gắn kết mạnh nhất là giữa `sadness` và `fear`. Các cảm xúc hiếm (như `anger`, `surprise`) gần như cô lập hoàn toàn.

---

## 5. Phân bổ Độ Dài Câu (Text Length)

![Text Length Distribution](../docs/images/06_text_length_distribution.png)
![Text Length by Emotion](../docs/images/07_text_length_by_emotion.png)

- Phân bổ Text Length không đổi so với RAW data. Độ dài trung bình: 15-25 từ. Cấu hình `max_length = 256` token là hoàn toàn phù hợp và an toàn.

---