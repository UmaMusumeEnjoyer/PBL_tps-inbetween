# -*- coding: utf-8 -*-
"""
scripts/download_weights.py
Công cụ tự động ghép mảnh và giải nén trọng số checkpoint từ repo,
hoặc tải trực tiếp từ GitHub Releases nếu có URL.
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import glob

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

REPO_OWNER = "UmaMusumeEnjoyer"
REPO_NAME = "PBL_tps-inbetween"

CHECKPOINT_REGISTRY = {
    "v3.0": {
        "pt_filename": "model_tps_v3.0_best.pt",
        "archive_filename": "model_tps_v3.0_best.zip",
        "description": "Thế hệ V3.0 (Rotation & Occlusion Aware) - Huấn luyện trên Golden Dataset V1_Patch3",
        "url": f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/v3.0/model_tps_v3.0_best.zip"
    },
    "v2.6": {
        "pt_filename": "model_tps_v2.6_best.pt",
        "archive_filename": "model_tps_v2.6_best.zip",
        "description": "Thế hệ V2.6 (Model Soup Calibrated) - Chuẩn vàng nét mảnh không bóng ma",
        "url": f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/v2.6/model_tps_v2.6_best.zip"
    },
    "v1.1": {
        "pt_filename": "model_tps_v1.1_production.pt",
        "archive_filename": "model_tps_v1.1_production.zip",
        "description": "Thế hệ V1.1 (Deep Ghost Erasure) - Ứng viên sản phẩm tối ưu khử bóng ma sâu",
        "url": f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/v1.1/model_tps_v1.1_production.zip"
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


def assemble_parts_and_extract(parts, target_zip, output_dir, target_pt):
    """Ghép các file .part01, .part02... thành file .zip rồi giải nén ra file .pt."""
    print(f"[*] Phát hiện {len(parts)} phân mảnh: {[os.path.basename(p) for p in parts]}")
    print(f"[*] Đang tự động ghép mảnh thành: {target_zip}...")
    with open(target_zip, 'wb') as outfile:
        for part in sorted(parts):
            with open(part, 'rb') as infile:
                outfile.write(infile.read())
    print(f"[✓] Ghép mảnh thành công ({os.path.getsize(target_zip) / (1024*1024):.2f} MB)!")

    print(f"[*] Đang giải nén file checkpoint...")
    with zipfile.ZipFile(target_zip, 'r') as zip_ref:
        namelist = zip_ref.namelist()
        zip_ref.extractall(output_dir)

    # Đổi tên file nếu tên trong zip khác target_pt
    extracted_pt = os.path.join(output_dir, namelist[0])
    if extracted_pt != target_pt and os.path.exists(extracted_pt):
        if os.path.exists(target_pt):
            os.remove(target_pt)
        os.rename(extracted_pt, target_pt)

    print(f"[✓] ĐÃ SẴN SÀNG: {target_pt} ({os.path.getsize(target_pt) / (1024*1024):.2f} MB)")

    # Dọn dẹp file zip tạm
    if os.path.exists(target_zip):
        os.remove(target_zip)
    return True


def main():
    parser = argparse.ArgumentParser(description="Tải hoặc giải nén trọng số mô hình TPS-Inbetween cho Anime 2D")
    parser.add_argument("--version", type=str, default="v3.0", choices=list(CHECKPOINT_REGISTRY.keys()),
                        help="Phiên bản trọng số muốn giải nén/tải (mặc định: v3.0)")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Thư mục lưu trữ trọng số (mặc định: checkpoints)")
    args = parser.parse_args()

    info = CHECKPOINT_REGISTRY[args.version]
    os.makedirs(args.output_dir, exist_ok=True)
    target_pt = os.path.join(args.output_dir, info["pt_filename"])
    target_zip = os.path.join(args.output_dir, info["archive_filename"])

    print("=" * 70)
    print("📦 BỘ TRÍCH XUẤT / TẢI TRỌNG SỐ TPS-INBETWEEN ANIME 2D")
    print(f"[*] Phiên bản    : {args.version}")
    print(f"[*] Mô tả        : {info['description']}")
    print(f"[*] Tệp mục tiêu : {info['pt_filename']}")
    print("=" * 70)

    # 1. Nếu file .pt đã tồn tại sẵn
    if os.path.exists(target_pt):
        size_mb = os.path.getsize(target_pt) / (1024 * 1024)
        print(f"[✓] Checkpoint đã tồn tại sẵn ({size_mb:.2f} MB). Bạn có thể sử dụng ngay!\n")
        return

    # 2. Kiểm tra nếu có sẵn các file phân mảnh .part* ngay trong thư mục checkpoints/
    part_pattern = os.path.join(args.output_dir, f"{info['archive_filename']}.part*")
    parts = glob.glob(part_pattern)
    if parts:
        assemble_parts_and_extract(parts, target_zip, args.output_dir, target_pt)
        print("\n[🎉] Hoàn tất! Bạn đã có thể chạy demo_infer.py hoặc server backend.\n")
        return

    # 3. Kiểm tra nếu có sẵn file .zip đầy đủ
    if os.path.exists(target_zip):
        print(f"[*] Tìm thấy file nén {target_zip}. Đang giải nén...")
        with zipfile.ZipFile(target_zip, 'r') as zip_ref:
            zip_ref.extractall(args.output_dir)
        print(f"[✓] ĐÃ SẴN SÀNG: {target_pt}")
        return

    # 4. Thử tải từ GitHub Releases
    url = info["url"]
    print(f"[*] Đang kết nối tải từ GitHub Releases: {url}")
    try:
        urllib.request.urlretrieve(url, target_zip, reporthook=download_progress_hook)
        print(f"\n[✓] Đã tải xong gói nén. Đang tiến hành giải nén...")
        with zipfile.ZipFile(target_zip, 'r') as zip_ref:
            zip_ref.extractall(args.output_dir)
        print(f"[✓] Đã giải nén thành công: {target_pt}")
        if os.path.exists(target_zip):
            os.remove(target_zip)
    except Exception as e:
        print(f"\n[-] Không thể tải từ GitHub Releases: {e}")
        print("\n👉 HƯỚNG DẪN DỰ PHÒNG CHO THÀNH VIÊN NHÓM:")
        print(f"   Repo git đã có sẵn các file phân mảnh trong thư mục '{args.output_dir}/'.")
        print(f"   Trên Linux/macOS, bạn có thể chạy lệnh nối file thủ công:")
        print(f"       cat {args.output_dir}/{info['archive_filename']}.part* > {target_zip}")
        print(f"       unzip {target_zip} -d {args.output_dir}/ && rm {target_zip}\n")


if __name__ == "__main__":
    main()
