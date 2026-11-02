# -*- coding: utf-8 -*-
"""
demo_infer.py
Kịch bản chạy thử nghiệm nội suy nhanh cho thành viên nhóm hoặc kiểm tra môi trường.
Cách chạy:
    python demo_infer.py --checkpoint checkpoints/model_tps_v3.0_best.pt --t 0.5
"""

import os
import argparse
from tps_anime import TPSPredictor


def main():
    parser = argparse.ArgumentParser(description="TPS-Inbetween Anime 2D Inference Demo")
    parser.add_argument("--frame0", type=str, default="demo_assets/frame_0.png", help="Đường dẫn Keyframe 0")
    parser.add_argument("--frame1", type=str, default="demo_assets/frame_1.png", help="Đường dẫn Keyframe 1")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model_tps_v3.0_best.pt", help="Đường dẫn file .pt")
    parser.add_argument("--output", type=str, default="demo_assets/output_inbetween.png", help="Đường dẫn lưu kết quả")
    parser.add_argument("--t", type=float, default=0.5, help="Mốc thời gian nội suy t in [0, 1]")
    parser.add_argument("--device", type=str, default=None, help="Thiết bị tính toán ('cuda' hoặc 'cpu')")
    parser.add_argument("--steps", type=int, default=1, help="Số khung hình muốn sinh đều giữa 2 keyframe (nếu > 1)")

    args = parser.parse_args()

    if not os.path.exists(args.frame0) or not os.path.exists(args.frame1):
        print(f"[-] Không tìm thấy ảnh đầu vào: {args.frame0} hoặc {args.frame1}")
        return

    if not os.path.exists(args.checkpoint):
        print(f"[-] Checkpoint '{args.checkpoint}' chưa được tải.")
        print(f"[*] Vui lòng chạy lệnh: python scripts/download_weights.py --version v3.0")
        return

    print("=" * 70)
    print("🚀 KHỞI ĐỘNG TPS-INBETWEEN ANIME 2D INFERENCE ENGINE")
    print(f"[*] Checkpoint : {args.checkpoint}")
    print(f"[*] Keyframe 0 : {args.frame0}")
    print(f"[*] Keyframe 1 : {args.frame1}")
    print("=" * 70)

    predictor = TPSPredictor(checkpoint_path=args.checkpoint, device=args.device)

    if args.steps > 1:
        print(f"[*] Đang sinh chuỗi {args.steps} khung hình liên tiếp...")
        seq = predictor.interpolate_sequence(args.frame0, args.frame1, steps=args.steps)
        out_dir = os.path.splitext(args.output)[0] + "_sequence"
        os.makedirs(out_dir, exist_ok=True)
        for i, (t_val, frame) in enumerate(seq):
            save_path = os.path.join(out_dir, f"frame_{i+1:02d}_t{t_val:.2f}.png")
            frame.save(save_path)
            print(f"    [+] Đã lưu: {save_path}")
        print(f"\n[✓] Hoàn thành sinh chuỗi khung hình tại: {out_dir}/")
    else:
        print(f"[*] Đang nội suy khung hình tại mốc t = {args.t:.2f}...")
        result = predictor.interpolate(args.frame0, args.frame1, t=args.t)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        result.save(args.output)
        print(f"\n[✓] Hoàn thành xuất sắc! Đã lưu ảnh kết quả: {args.output}")


if __name__ == "__main__":
    main()
