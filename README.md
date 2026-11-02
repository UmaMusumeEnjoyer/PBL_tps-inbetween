# 🎬 TPS-Inbetween Anime 2D — Inference Core SDK

> **Dành cho AI Agent & Developer:** Tài liệu này cung cấp toàn bộ bối cảnh hệ thống (Architecture Context), quy chuẩn dữ liệu I/O, giải pháp xử lý lỗi kiến trúc và hướng dẫn tích hợp trực tiếp vào Backend (FastAPI, Flask, Celery, Desktop App).

---

## 📌 1. Bối Cảnh Dự Án (System Context)

### Bài toán giải quyết
Trong hoạt họa Anime 2D truyền thống, họa sĩ vẽ các khung hình khóa chính (**Keyframes**) tại thời điểm $t=0$ và $t=1$. Khâu vẽ các khung hình trung gian (**Inbetween Frames**) tại thời điểm $t \in (0, 1)$ tốn rất nhiều công sức. 

Mô hình gốc **TPS-Inbetween** (huấn luyện trên dữ liệu 3D Mixamo) khi áp dụng vào Anime 2D gặp các vấn đề nghiêm trọng:
1. **Ghosting (Bóng ma):** Do cơ chế hòa trộn tuyến tính cứng $(1-t)I_0 + tI_1 + \text{res}$, khi các đường nét chuyển động lệch nhau sẽ sinh ra 2 bóng mờ song song.
2. **Stroke Thickening & Blurring:** Mất chi tiết nét vẽ mảnh (tóc, viền mắt, nếp gấp vải).
3. **Occlusion & Rotation Breakdown:** Khi nhân vật quay mặt/người 3D, các chi tiết bị che khuất sẽ bị xé rách hoặc nối sai topology.

### Cải tiến trong Repository này
Repo này chứa **Inference Engine** đã qua huấn luyện & tinh chỉnh chuyên sâu trên **Sakuga Golden Dataset** (bộ dữ liệu hoạt họa Nhật Bản chất lượng cao):
- **ASF-GR Engine (Adaptive Stroke Fusion & Gated Residual):** Tự động học ma trận hòa trộn thích ứng theo thời gian $t$, triệt tiêu hiện tượng bóng ma.
- **Occlusion-Aware & Flow Propagator (Thế hệ V3.0):** Tích hợp mạng phân tích bất đối xứng đặc trưng (`AsymmetricFeatureAnalyzer`), cổng hiển thị kép (`AsymmetricStrokeFusionHead`) và bộ dẫn hướng luồng biên bao (`ContourGuidedFlowPropagator`), giải quyết triệt để góc quay 3D và tóc/vải bay.
- **Smart Model Dispatcher:** Tự động phát hiện phiên bản checkpoint và nạp đúng cấu trúc mạng mà không cần cấu hình thủ công.

---

## 🏗️ 2. Kiến Trúc Mạng & Bảng Khớp State Dict (Architecture Spec)

> [!IMPORTANT]
> **Lý do trước đây server bị "nổ" (Crash / Missing keys):**
> File checkpoint `.pt` được huấn luyện với các module bổ sung (`asf_head`, `asym_analyzer`, `contour_propagator` và `unet` 2 đầu ra `gate_erase` + `detail_add`). Nếu nạp vào class `TPS-Inbetween` của tác giả gốc sẽ bị lỗi `RuntimeError: Missing/Unexpected key(s)`.
> 
> **Trong repo này:** Hàm `build_tps_model()` và class `TPSPredictor` đã được thiết kế sẵn để **tự động khớp bit-exact 100%** với state dict của từng thế hệ.

### Bảng đối chiếu các thế hệ Checkpoint:

| Thế hệ | Checkpoint Khuyên Dùng | Module Cốt Lõi | Đặc điểm & Ứng dụng |
|:---|:---|:---|:---|
| **V3.0 (Target)** | `model_tps_v3.0_best.pt` | `Contextnet`, `IFNet`, `asym_analyzer`, `contour_propagator_v30`, `asf_head_v30`, `SynthesisUnet_v30` | **Tối ưu toàn diện nhất:** Xử lý che khuất, góc xoay 3D, tóc/vải bay, khôi phục nét vẽ chi tiết. |
| **V2.6 (Soup)** | `model_tps_v2.6_best.pt` | `Contextnet`, `IFNet`, `asf_head_v26`, `SynthesisUnet_v26` | **Chuẩn vàng độ mảnh nét:** Cân bằng mật độ mực tuyệt đối, nền trắng sạch 100%. |
| **V1.1 (Prod)** | `model_tps_v1.1_production.pt` | `Contextnet`, `IFNet`, `asf_head`, `SynthesisUnet_v1` | Tối ưu khử bóng ma sâu trên chuỗi cảnh tĩnh hoặc chuyển động tịnh tiến đơn giản. |

---

## 📥 3. Đặc Tả Dữ Liệu Đầu Vào & Đầu Ra (I/O Specification)

### Quy chuẩn Nét vẽ Anime 2D (Lineart Standard)
- **Đầu vào hiển thị (Human / File format):**
  - **Nền (Background):** Màu trắng tuyệt đối (`RGB: 255, 255, 255` hoặc Grayscale `255`).
  - **Nét vẽ (Strokes):** Màu đen (`RGB: 0, 0, 0` hoặc Grayscale `0`).
- **Không gian tính toán bên trong mô hình (Internal Tensor):**
  - Mô hình tính toán trên tensor chuẩn hóa: **Nét vẽ = 1.0, Nền = 0.0**.
  - `TPSPredictor` đã **tự động đảo ngược (`invert=True`)** lúc nạp ảnh và hoàn trả lại đúng ảnh **Nền trắng / Nét đen** khi xuất kết quả. Agent/Developer không cần xử lý thủ công!

### Kích thước & Padding
- Mạng UNet yêu cầu kích thước ảnh chia hết cho 32 (hoặc chuẩn độ phân giải Anime 540p: $960 \times 540$ padding thành $960 \times 544$).
- `TPSPredictor` tự động thực hiện replicate padding trước khi forward và unpad về đúng kích thước gốc của keyframe trước khi trả về.

---

## 🚀 4. Hướng Dẫn Tích Hợp Cho Backend / API Agent

### 4.1. Cài đặt môi trường
Khuyến nghị Python 3.9 - 3.11:
```bash
git clone <URL_REPO_CUA_BAN>
cd tps-inbetween-anime

pip install -r requirements.txt
```

### 4.2. Tải hoặc đặt Checkpoint
Đặt file `.pt` vào thư mục `checkpoints/` (ví dụ `checkpoints/model_tps_v3.0_best.pt`):
```bash
python scripts/download_weights.py --version v3.0
```

### 4.3. Code mẫu tích hợp vào FastAPI Service

```python
import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from PIL import Image
from tps_anime import TPSPredictor

app = FastAPI(title="Anime 2D Inbetweening Service")

# Khởi tạo predictor 1 lần duy nhất khi worker khởi động (Singleton)
# Tự động chọn CUDA nếu có GPU, fallback về CPU nếu không có
predictor = TPSPredictor(
    checkpoint_path="checkpoints/model_tps_v3.0_best.pt",
    device=None  # Hoặc chỉ định "cuda:0" / "cpu"
)

@app.post("/api/v1/interpolate")
async def interpolate_frames(
    frame0: UploadFile = File(..., description="Keyframe bắt đầu (PNG/JPG)"),
    frame1: UploadFile = File(..., description="Keyframe kết thúc (PNG/JPG)"),
    t: float = Form(0.5, ge=0.0, le=1.0, description="Thời điểm nội suy t in [0, 1]")
):
    try:
        img0 = Image.open(io.BytesIO(await frame0.read()))
        img1 = Image.open(io.BytesIO(await frame1.read()))

        # Suy luận sinh khung hình trung gian
        result_img = predictor.interpolate(img0, img1, t=t)

        # Trả về ảnh PNG
        buffer = io.BytesIO()
        result_img.save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/interpolate-sequence")
async def interpolate_sequence(
    frame0: UploadFile = File(...),
    frame1: UploadFile = File(...),
    steps: int = Form(3, ge=1, le=12, description="Số frame muốn sinh ở giữa (VD: steps=3 -> 4x framerate)")
):
    """Sinh chuỗi khung hình phục vụ On-twos, On-threes hoặc tăng fps"""
    img0 = Image.open(io.BytesIO(await frame0.read()))
    img1 = Image.open(io.BytesIO(await frame1.read()))

    sequence = predictor.interpolate_sequence(img0, img1, steps=steps)
    # sequence là list gồm các tuple: [(t_1, PIL.Image), (t_2, PIL.Image), ...]
    return {"message": f"Successfully generated {len(sequence)} inbetween frames"}
```

---

## ⚡ 5. Kiểm Thử Nhanh Bằng CLI (Command Line Interface)

Chạy thử nghiệm 1 cặp ảnh mẫu có sẵn trong thư mục `demo_assets/`:

```bash
# Sinh 1 khung hình tại mốc t = 0.5
python demo_infer.py --checkpoint checkpoints/model_tps_v3.0_best.pt --t 0.5

# Sinh chuỗi 3 khung hình liên tiếp (t = 0.25, 0.50, 0.75)
python demo_infer.py --checkpoint checkpoints/model_tps_v3.0_best.pt --steps 3
```
Kết quả ảnh sẽ được ghi vào thư mục `demo_assets/`.

---

## 🛠️ 6. Sơ Đồ Cấu Trúc Mã Nguồn

```plaintext
tps-inbetween-anime/
├── tps_anime/
│   ├── __init__.py           # Khai báo public API: TPSPredictor, build_tps_model
│   ├── architectures.py      # Định nghĩa đầy đủ các module mạng (ASF-GR, Asymmetric, Propagator)
│   ├── tps_solver.py         # Giải thuật TPS Solver 8 điểm neo biên (chống méo rách khung hình)
│   ├── predictor.py          # Wrapper TPSPredictor cấp cao (quản lý I/O, padding, inference)
│   └── utils.py              # Xử lý chuẩn hóa tensor, chuyển đổi ảnh PIL <-> Tensor
├── checkpoints/              # Nơi lưu file weights .pt (bị ignore bởi git, tải qua script)
├── demo_assets/              # Ảnh keyframe mẫu anime 540p phục vụ kiểm thử
│   ├── frame_0.png
│   └── frame_1.png
├── scripts/
│   └── download_weights.py   # Script tự động tải checkpoint từ Release URL
├── demo_infer.py             # Script CLI chạy thử nghiệm end-to-end
├── requirements.txt          # Thư viện tối thiểu (torch, torchvision, opencv, pillow, scipy)
├── .gitignore                # Loại trừ file weights nặng và cache
└── README.md                 # Tài liệu này
```

---

## 🤖 7. Nguyên Tắc Cho AI Agent Khi Mở Rộng / Sửa Đổi Code

1. **Tuyệt đối không sửa đổi tên các layer con** trong `architectures.py` (như `conv_erase`, `conv_add`, `asf_head`, `asym_analyzer`, `contour_propagator`), vì việc này sẽ làm gãy cấu trúc `state_dict` đã lưu trong file checkpoint `.pt`.
2. **Khi thêm API hoặc Endpoint mới:** Luôn tận dụng `TPSPredictor` từ `tps_anime` thay vì tự load mô hình thủ công.
3. **Khi cần tăng tốc:** Có thể bật `torch.cuda.amp.autocast()` trong môi trường GPU hoặc export sang TensorRT/ONNX (chú ý toán tử grid_sample).
