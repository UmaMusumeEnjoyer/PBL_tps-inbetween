# -*- coding: utf-8 -*-
"""
tps_anime/predictor.py
Wrapper cấp cao phục vụ tích hợp trực tiếp vào Backend (FastAPI, Flask, Celery, GUI).
Đơn giản hóa quá trình tiền xử lý, tính toán TPS flow và xuất khung hình inbetween.
"""

import os
import torch
from PIL import Image

from .architectures import build_tps_model
from .utils import load_image_to_tensor, tensor_to_image, pad_to_multiple, unpad


class TPSPredictor:
    """
    Inference Engine phục vụ sinh khung hình trung gian (Inbetween Frame) từ 2 Keyframe.
    """
    def __init__(self, checkpoint_path=None, version="v3.0", device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.version = version
        self.checkpoint_path = checkpoint_path
        self.model = build_tps_model(
            checkpoint_path=self.checkpoint_path,
            version=self.version,
            map_location=self.device
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def interpolate(self, frame_0, frame_1, t=0.5, kps=None, pad_multiple=32):
        """
        Nội suy khung hình tại mốc thời gian t in (0, 1).
        
        Args:
            frame_0: Đường dẫn file, numpy array hoặc PIL.Image (Keyframe đầu).
            frame_1: Đường dẫn file, numpy array hoặc PIL.Image (Keyframe cuối).
            t: Mốc thời gian trung gian (float từ 0.0 đến 1.0). Mặc định là 0.5.
            kps: Tọa độ keypoints tương ứng nếu có sẵn (np.ndarray hình dạng [2, N, 2]).
            pad_multiple: Bội số padding cho mạng UNet (mặc định 32).
            
        Returns:
            PIL.Image (Khung hình nét anime tại thời điểm t).
        """
        assert 0.0 <= t <= 1.0, f"Tham số thời gian t phải nằm trong [0.0, 1.0], nhận giá trị: {t}"

        x0 = load_image_to_tensor(frame_0, invert=True)
        x1 = load_image_to_tensor(frame_1, invert=True)

        # Đảm bảo 2 ảnh cùng kích thước
        if x0.shape != x1.shape:
            h, w = x0.shape[2], x0.shape[3]
            x1 = torch.nn.functional.interpolate(x1, size=(h, w), mode='bilinear', align_corners=False)

        x0_pad, pad_box = pad_to_multiple(x0, multiple=pad_multiple)
        x1_pad, _ = pad_to_multiple(x1, multiple=pad_multiple)

        x0_pad = x0_pad.to(self.device)
        x1_pad = x1_pad.to(self.device)
        t_tensor = torch.tensor([float(t)], dtype=torch.float32, device=self.device)

        kps_input = [kps] if kps is not None else None

        outputs = self.model(x0_pad, x1_pad, t_tensor, kps_input=kps_input)
        if isinstance(outputs, tuple):
            pred_t = outputs[0]
        else:
            pred_t = outputs

        pred_unpad = unpad(pred_t, pad_box)
        result_img = tensor_to_image(pred_unpad, invert=True)
        return result_img

    def interpolate_sequence(self, frame_0, frame_1, steps=3, kps=None):
        """
        Nội suy chuỗi nhiều khung hình liên tiếp giữa 2 keyframe.
        Ví dụ: steps=3 sinh ra các khung hình tại t = 0.25, 0.5, 0.75.
        """
        frames = []
        for i in range(1, steps + 1):
            t_val = i / (steps + 1)
            f = self.interpolate(frame_0, frame_1, t=t_val, kps=kps)
            frames.append((t_val, f))
        return frames
