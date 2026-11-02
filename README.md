# 🎬 TPS-Inbetween for Anime 2D (Inference Core)

Thư viện lõi và mô hình suy luận sinh khung hình trung gian (Inbetween Frames) nét vẽ Anime 2D tại mốc thời gian $t \in [0, 1]$ bất kỳ, giải quyết triệt để vấn đề méo nét, phình nét và bóng ma (ghosting).

Kho mã nguồn này được đóng gói sẵn để **tích hợp trực tiếp vào Backend (FastAPI, Flask), Celery worker hoặc Desktop App** mà không gặp bất kỳ lỗi xung đột hay lệch kiến trúc nào giữa các thế hệ checkpoint.

---

## 🌟 Các Tính Năng Nổi Bật

- **Tự động thích ứng kiến trúc (Smart Architecture Dispatcher):** Nhận diện chính xác cấu trúc checkpoint từ V1.0/V1.1, V2.6 (Model Soup Gold) đến V3.0 (Rotation & Occlusion Aware) và V3.2.
- **Wrapper Cấp Cao (`TPSPredictor`):** Chỉ cần truyền vào 2 ảnh (PIL Image hoặc đường dẫn file) và thời điểm $t$, tự động xử lý chuẩn hóa nền trắng/nét đen, padding bội số UNet và unpadding chuẩn kích thước gốc.
- **Hỗ trợ sinh chuỗi chuyển động:** Hàm `interpolate_sequence()` hỗ trợ sinh On-twos, On-threes hoặc 24 FPS mượt mà.
- **Không dính mã nguồn thừa:** Tách biệt hoàn toàn khỏi các script cào dữ liệu, pipeline training nặng nề trên Kaggle.

---

## 🚀 Cài Đặt (Quick Installation)

Khuyến nghị sử dụng Python 3.9 - 3.11 trong môi trường ảo:

```bash
git clone https://github.com/YOUR_ORGANIZATION_OR_USERNAME/tps-inbetween-anime.git
cd tps-inbetween-anime

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

---

## 📦 Tải Trọng Số Mô Hình (Weights)

Vì file trọng số `.pt` (~40.5MB) được lưu trữ trên **GitHub Releases / Cloud**, bạn hãy tải checkpoint về thư mục `checkpoints/`:

```bash
# Tải phiên bản V3.0 (Rotation & Occlusion Aware - Khuyên dùng)
python scripts/download_weights.py --version v3.0

# Hoặc tải phiên bản V2.6 (Chuẩn vàng nét mảnh)
python scripts/download_weights.py --version v2.6
```
*(Nếu chưa cấu hình link release, bạn chỉ cần sao chép file `.pt` trực tiếp vào thư mục `checkpoints/`)*.

---

## 💻 Hướng Dẫn Sử Dụng (Quickstart)

### 1. Chạy thử nghiệm ngay bằng CLI
```bash
python demo_infer.py --checkpoint checkpoints/model_tps_v3.0_best.pt --t 0.5
```
Ảnh kết quả sẽ được tạo tại `demo_assets/output_inbetween.png`.

---

### 2. Tích hợp vào Backend / Python Code (3 Dòng Code)

```python
from tps_anime import TPSPredictor

# 1. Khởi tạo Predictor (tự nhận diện GPU/CPU và cấu trúc model)
predictor = TPSPredictor(checkpoint_path="checkpoints/model_tps_v3.0_best.pt")

# 2. Sinh khung hình tại mốc thời gian t = 0.5
inbetween_frame = predictor.interpolate("demo_assets/frame_0.png", "demo_assets/frame_1.png", t=0.5)

# 3. Lưu hoặc trả về qua API
inbetween_frame.save("result_frame.png")
```

### 3. Sinh chuỗi khung hình liên tiếp
```python
# Sinh 3 khung hình trung gian (t = 0.25, 0.50, 0.75)
sequence = predictor.interpolate_sequence("frame_0.png", "frame_1.png", steps=3)

for t_val, frame in sequence:
    frame.save(f"frame_t_{t_val:.2f}.png")
```

---

## 📂 Cấu Trúc Thư Mục Repository

```plaintext
tps-inbetween-anime/
├── tps_anime/                          # Package Python chính
│   ├── __init__.py                     # Expose: TPSPredictor, build_tps_model
│   ├── architectures.py                # Định nghĩa kiến trúc (ASF-GR, Asymmetric Head, Propagator)
│   ├── tps_solver.py                   # Giải thuật TPS Solver 8 điểm neo biên
│   ├── predictor.py                    # Wrapper xử lý inference cấp cao
│   └── utils.py                        # Xử lý padding, load/save ảnh nét anime
├── checkpoints/                        # Nơi đặt file weights .pt (bị ignore bởi git)
├── demo_assets/                        # Cặp keyframe mẫu phục vụ kiểm thử
├── scripts/
│   └── download_weights.py             # Script tự động kéo model weights
├── demo_infer.py                       # CLI demo chạy thử
├── requirements.txt                    # Thư viện phụ thuộc tối thiểu
├── .gitignore                          # Chặn commit file nặng và file rác
└── README.md                           # Hướng dẫn sử dụng
```

---

## 👥 Dành cho Thành viên Nhóm (Team Integration)

Mọi thành viên làm phần Backend API, Celery worker hay Frontend desktop chỉ cần:
1. Clone repo này về máy hoặc cài đặt dưới dạng submodule: `git submodule add <repo_url>`.
2. Import `from tps_anime import TPSPredictor` và gọi suy luận.
3. Toàn bộ trọng số state_dict đã được khớp bit-exact với các class mạng, triệt tiêu 100% nguy cơ lỗi kiến trúc.
