# -*- coding: utf-8 -*-
"""
scripts/download_weights.py
Công cụ tự động tải trọng số checkpoint cho các thành viên trong nhóm.
Hỗ trợ tải trực tiếp từ GitHub Releases hoặc Google Drive / Hugging Face.
"""

import os
import sys
import argparse
import urllib.request

# Cấu hình danh sách phiên bản và link tải (Cập nhật URL Release khi tạo xong GitHub Release)
CHECKPOINT_REGISTRY = {
    "v3.0": {
        "filename": "model_tps_v3.0_best.pt",
        "description": "Thế hệ V3.0 (Rotation & Occlusion Aware) - Huấn luyện trên Golden Dataset V1_Patch3",
        "url": "https://github.com/YOUR_ORGANIZATION_OR_USERNAME/tps-inbetween-anime/releases/download/v3.0/model_tps_v3.0_best.pt"
    },
    "v2.6": {
        "filename": "model_tps_v2.6_best.pt",
        "description": "Thế hệ V2.6 (Model Soup Calibrated) - Chuẩn vàng nét mảnh không bóng ma",
        "url": "https://github.com/YOUR_ORGANIZATION_OR_USERNAME/tps-inbetween-anime/releases/download/v2.6/model_tps_v2.6_best.pt"
    },
    "v1.1": {
        "filename": "model_tps_v1.1_production.pt",
        "description": "Thế hệ V1.1 (Deep Ghost Erasure) - Ứng viên sản phẩm tối ưu khử bóng ma sâu",
        "url": "https://github.com/YOUR_ORGANIZATION_OR_USERNAME/tps-inbetween-anime/releases/download/v1.1/model_tps_v1.1_production.pt"
    }
}


def download_progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r[*] Đang tải: {percent:5.1f}% [{mb_down:6.1f}MB / {mb_total:6.1f}MB]")
        sys.stdout.flush()
    else:
        mb_down = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r[*] Đang tải: {mb_down:6.1f}MB")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Tải trọng số mô hình TPS-Inbetween cho Anime 2D")
    parser.add_argument("--version", type=str, default="v3.0", choices=list(CHECKPOINT_REGISTRY.keys()),
                        help="Phiên bản trọng số muốn tải (mặc định: v3.0)")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Thư mục lưu trữ trọng số (mặc định: checkpoints)")
    args = parser.parse_args()

    info = CHECKPOINT_REGISTRY[args.version]
    os.makedirs(args.output_dir, exist_ok=True)
    target_path = os.path.join(args.output_dir, info["filename"])

    print("=" * 70)
    print("📦 BỘ TẢI TRỌNG SỐ TPS-INBETWEEN ANIME 2D")
    print(f"[*] Phiên bản    : {args.version}")
    print(f"[*] Mô tả        : {info['description']}")
    print(f"[*] Tên tệp      : {info['filename']}")
    print(f"[*] Nơi lưu      : {target_path}")
    print("=" * 70)

    if os.path.exists(target_path):
        size_mb = os.path.getsize(target_path) / (1024 * 1024)
        print(f"[!] Checkpoint đã tồn tại sẵn ({size_mb:.2f} MB). Bỏ qua lượt tải.")
        return

    url = info["url"]
    if "YOUR_ORGANIZATION_OR_USERNAME" in url:
        print("\n[!] LƯU Ý CHO THÀNH VIÊN NHÓM:")
        print("    Link GitHub Release chưa được cập nhật chính thức.")
        print(f"    Bạn có thể sao chép file '{info['filename']}' thủ công vào thư mục '{args.output_dir}/'")
        print("    hoặc cập nhật URL chính xác trong file scripts/download_weights.py.\n")
        return

    print(f"[*] Đang kết nối tới máy chủ tải...")
    try:
        urllib.request.urlretrieve(url, target_path, reporthook=download_progress_hook)
        print(f"\n[✓] Đã tải hoàn tất: {target_path}")
    except Exception as e:
        print(f"\n[-] Lỗi trong quá trình tải: {e}")
        if os.path.exists(target_path):
            os.remove(target_path)


if __name__ == "__main__":
    main()
