# -*- coding: utf-8 -*-
"""
tps_anime/utils.py
Tiện ích xử lý ảnh đầu vào và đầu ra cho TPS-Inbetween.
"""

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def load_image_to_tensor(image_input, target_size=None, invert=True):
    """
    Nạp ảnh từ đường dẫn file, numpy array hoặc PIL.Image thành Tensor (1, 1, H, W).
    Chuẩn hóa ảnh nét anime: Nền trắng (1.0), Nét đen (0.0).
    Mặc định invert=True để chuyển sang không gian tính toán: Nét vẽ = 1.0, Nền = 0.0.
    """
    if isinstance(image_input, str):
        img_pil = Image.open(image_input).convert('L')
    elif isinstance(image_input, Image.Image):
        img_pil = image_input.convert('L')
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 3:
            img_pil = Image.fromarray(image_input).convert('L')
        else:
            img_pil = Image.fromarray(image_input)
    elif isinstance(image_input, torch.Tensor):
        if image_input.ndim == 2:
            image_input = image_input.unsqueeze(0).unsqueeze(0)
        elif image_input.ndim == 3:
            image_input = image_input.unsqueeze(0)
        return image_input.float()
    else:
        raise ValueError(f"Định dạng ảnh không được hỗ trợ: {type(image_input)}")

    if target_size is not None:
        img_pil = img_pil.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)

    img_arr = np.array(img_pil, dtype=np.float32) / 255.0

    if invert:
        # Nét đen (0) -> 1.0, Nền trắng (255) -> 0.0
        img_arr = 1.0 - img_arr

    tensor = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0).float()
    return tensor


def tensor_to_image(tensor, invert=True):
    """
    Chuyển đổi tensor (1, 1, H, W) kết quả về PIL Image (Grayscale L).
    """
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0).squeeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.squeeze(0)

    arr = tensor.detach().cpu().float().numpy()
    arr = np.clip(arr, 0.0, 1.0)

    if invert:
        # Chuyển ngược lại Nét đen (0), Nền trắng (255)
        arr = 1.0 - arr

    arr_u8 = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr_u8, mode='L')


def pad_to_multiple(tensor, multiple=32):
    """
    Pad ảnh để chiều cao và chiều rộng chia hết cho multiple (phục vụ UNet downsample).
    Trả về: tensor_padded, (pad_left, pad_right, pad_top, pad_bottom)
    """
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
    return padded, (pad_left, pad_right, pad_top, pad_bottom)


def unpad(tensor, pad_box):
    """
    Cắt bỏ phần padding để khôi phục kích thước ảnh gốc.
    """
    pad_left, pad_right, pad_top, pad_bottom = pad_box
    h = tensor.shape[2]
    w = tensor.shape[3]
    return tensor[:, :, pad_top:h - pad_bottom, pad_left:w - pad_right]
