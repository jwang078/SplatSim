import cv2
import subprocess
import datetime
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "recordings"
FPS = 30

cap = cv2.VideoCapture(0)

# Force the stream width and height to match standard 1080p widescreen (16:9)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Optional but highly recommended for capture cards:
# Forces MJPG stream demuxing to keep processing fast and uncompressed buffer queues empty
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[+] Stream: {actual_width}x{actual_height} @ {FPS} FPS (recording)")
print("Press SPACE to start/stop recording, Q to quit")

def start_ffmpeg(width, height, fps, path):
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


ffmpeg_proc = None
record_path = None

while True:
    ret, frame = cap.read()
    if ret:
        if ffmpeg_proc is not None and ffmpeg_proc.stdin is not None:
            try:
                ffmpeg_proc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                print("[!] ffmpeg pipe broke — stopping recording")
                ffmpeg_proc = None

        display = cv2.resize(frame, (actual_width//2, actual_height//2), interpolation=cv2.INTER_LINEAR)
        if ffmpeg_proc is not None:
            cv2.circle(display, (30, 30), 12, (0, 0, 255), -1)
            cv2.putText(display, "REC", (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("HDMI Capture Stream", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        if ffmpeg_proc is None:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            record_path = OUTPUT_DIR / f"gopro_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            ffmpeg_proc = start_ffmpeg(actual_width, actual_height, FPS, record_path)
            print(f"[+] Recording started: {record_path}")
        else:
            if ffmpeg_proc.stdin is not None:
                ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
            print(f"[+] Recording saved: {record_path}")
            ffmpeg_proc = None
            record_path = None

if ffmpeg_proc is not None:
    if ffmpeg_proc.stdin is not None:
        ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    print(f"[+] Recording saved: {record_path}")

cap.release()
cv2.destroyAllWindows()