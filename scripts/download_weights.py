# -*- coding: utf-8 -*-
"""
scripts/download_weights.py
Công cụ tự động tải và giải nén trọng số checkpoint cho các thành viên trong nhóm.
Hỗ trợ tải trực tiếp từ GitHub Releases dạng ZIP/TAR.GZ hoặc file đơn.
"""

import os
import sys
import argparse
import urllib.request
import zipfile

CHECKPOINT_REGISTRY = {
    "v3.0": {
        "pt_filename": "model_tps_v3.0_best.pt",
        "archive_filename": "model_tps_v3.0_best.zip",
        "description": "Thế hệ V3.0 (Rotation & Occlusion Aware) - Huấn luyện trên Golden Dataset V1_Patch3",
        "url": "https://github.com/YOUR_ORGANIZATION_OR_USERNAME/tps-inbetween-anime/releases/download/v3.0/model_tps_v3.0_best.zip"
    },
    "v2.6": {
        "pt_filename": "model_tps_v2.6_best.pt",
        "archive_filename": "model_tps_v2.6_best.zip",
        "description": "Thế hệ V2.6 (Model Soup Calibrated) - Chuẩn vàng nét mảnh không bóng ma",
        "url": "https://github.com/YOUR_ORGANIZATION_OR_USERNAME/tps-inbetween-anime/releases/download/v2.6/model_tps_v2.6_best.zip"
    },
    "v1.1": {
        "pt_filename": "model_tps_v1.1_production.pt",
        "archive_filename": "model_tps_v1.1_production.zip",
        "description": "Thế hệ V1.1 (Deep Ghost Erasure) - Ứng viên sản phẩm tối ưu khử bóng ma sâu",
        "url": "https://github.com/YOUR_ORGANIZATION_OR_USERNAME/tps-inbetween-anime/releases/download/v1.1/model_tps_v1.1_production.zip"
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
    target_pt = os.path.join(args.output_dir, info["pt_filename"])
    target_zip = os.path.join(args.output_dir, info["archive_filename"])

    print("=" * 70)
    print("📦 BỘ TẢI TRỌNG SỐ TPS-INBETWEEN ANIME 2D")
    print(f"[*] Phiên bản    : {args.version}")
    print(f"[*] Mô tả        : {info['description']}")
    print(f"[*] Tệp mục tiêu : {info['pt_filename']}")
    print("=" * 70)

    if os.path.exists(target_pt):
        size_mb = os.path.getsize(target_pt) / (1024 * 1024)
        print(f"[!] Checkpoint đã tồn tại sẵn ({size_mb:.2f} MB). Bỏ qua lượt tải.")
        return

    url = info["url"]
    if "YOUR_ORGANIZATION_OR_USERNAME" in url:
        print("\n[!] LƯU Ý CHO THÀNH VIÊN NHÓM:")
        print("    Link GitHub Release chưa được cập nhật chính thức.")
        print(f"    Bạn có thể sao chép file '{info['pt_filename']}' (hoặc giải nén '{info['archive_filename']}')")
        print(f"    trực tiếp vào thư mục '{args.output_dir}/'.\n")
        return

    print(f"[*] Đang tải gói nén từ GitHub Releases: {url}")
    try:
        urllib.request.urlretrieve(url, target_zip, reporthook=download_progress_hook)
        print(f"\n[✓] Đã tải xong gói nén. Đang tiến hành giải nén...")
        with zipfile.ZipFile(target_zip, 'r') as zip_ref:
            zip_ref.extractall(args.output_dir)
        print(f"[✓] Đã giải nén thành công file: {target_pt}")
        if os.path.exists(target_zip):
            os.remove(target_zip)
    except Exception as e:
        print(f"\n[-] Lỗi trong quá trình tải/giải nén: {e}")
        if os.path.exists(target_zip):
            os.remove(target_zip)


if __name__ == "__main__":
    main()
