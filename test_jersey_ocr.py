"""
Test jersey number OCR on output video.
Usage:
    python test_jersey_ocr.py --video output_videos/output_enhanced.avi
    python test_jersey_ocr.py --video output_videos/download.mp4
"""
import argparse, os, sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ocr.jersey_ocr import read_jersey_number, crop_jersey_region


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='output_videos/download.mp4')
    parser.add_argument('--frame-interval', type=int, default=30,
                        help='Test every N frames')
    parser.add_argument('--max-frames', type=int, default=10,
                        help='Max frames to test')
    parser.add_argument('--show', action='store_true',
                        help='Display each crop with result')
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"File not found: {args.video}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total} frames")
    print("Detecting jersey numbers...")
    print()

    from ultralytics import YOLO
    model = YOLO('models/player_detector.pt')

    frame_nums = list(range(0, total, args.frame_interval))[:args.max_frames]
    all_results = []

    for fi in frame_nums:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.3, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            conf = float(box.conf[0])

            # Only read numbers on players
            if cls_name not in ('player',):
                continue

            bbox = [x1, y1, x2, y2]
            num = read_jersey_number(frame, bbox)

            crop = crop_jersey_region(frame, bbox)
            crop_h = crop.shape[0] if crop is not None else 0

            result = {
                'frame': fi,
                'class': cls_name,
                'conf': conf,
                'bbox': bbox,
                'crop_h': crop_h,
                'number': num,
            }
            all_results.append(result)

            if args.show and crop is not None:
                try:
                    cv2.imshow(f'Frame {fi} - Player', crop)
                    key = cv2.waitKey(0)
                    if key == ord('q'):
                        break
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass  # headless OpenCV

    cap.release()

    # Report
    detected = [r for r in all_results if r['number'] is not None]
    print(f"Total players checked: {len(all_results)}")
    print(f"Numbers detected: {len(detected)} "
          f"({len(detected)/max(1,len(all_results))*100:.0f}%)")
    print()

    if detected:
        print("Detected numbers:")
        for r in detected:
            print(f"  Frame {r['frame']:5d}  "
                  f"#{r['number']:2d}  "
                  f"(conf={r['conf']:.2f}, "
                  f"crop_h={r['crop_h']}px)")
    else:
        print("No jersey numbers detected.")
        print()
        print("Possible issues:")
        print("  - Crop region too small (try adjusting crop_jersey_region)")
        print("  - Player too far / low resolution")
        print("  - EasyOCR model not downloaded yet")
        print("  - Try: pip install easyocr")
        print()
        print("Sample crops:")

        for r in all_results[:5]:
            print(f"  Frame {r['frame']}  cls={r['class']}  "
                  f"bbox={r['bbox']}  crop_h={r['crop_h']}px")


if __name__ == '__main__':
    main()
