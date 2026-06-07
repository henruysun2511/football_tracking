import cv2
import os
import subprocess
import tempfile

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(frames, output_path, fps=24):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    # Save to temp .avi with no compression (fast)
    tmp = tempfile.NamedTemporaryFile(suffix='.avi', delete=False)
    tmp_path = tmp.name
    tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', tmp_path,
             '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
             '-movflags', '+faststart',
             output_path],
            capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: OpenCV mp4v
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for frame in frames:
            out.write(frame)
        out.release()
    finally:
        os.unlink(tmp_path)