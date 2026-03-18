# Experiments Directory

Thư mục này dùng để quản lý và theo dõi quá trình phát triển mô hình.

## Structure

```
experiments/
├── experiment_log.md        # Bảng tổng hợp kết quả tất cả thí nghiệm
├── configs/                 # Lưu snapshot config của mỗi lần chạy
│   ├── v1_roberta_baseline.yaml
│   └── v2_xlm_roberta_improved.yaml
└── README.md                # File này
```

## How to Use

1. **Trước khi chạy thí nghiệm mới:** Copy config YAML vào `experiments/configs/` với prefix `vN_`
2. **Sau khi chạy xong:** Update bảng trong `experiment_log.md` với kết quả
3. **Ghi chú quan trọng:** Viết changelog mô tả thay đổi so với version trước
