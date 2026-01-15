import cv2
import time
import os
from threading import Thread

# Import your custom detection module
try:
    import detect 
except ImportError:
    print("❌ Error: 'detect.py' not found.")
    exit()

# --- CONFIGURATION ---
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480

class WebcamStream:
    def __init__(self, src=0, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT):
        try:
            self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        except:
            self.stream = cv2.VideoCapture(src)

        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, 30)

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True
        self.stream.release()

def run_system(source):
    # 1. Setup Input Source
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    is_live = isinstance(source, int) or "rtsp" in str(source)
    
    if is_live:
        print(f"📡 Mode: LIVE CAMERA - Source: {source}")
        cap = WebcamStream(source).start()
        time.sleep(1.0) 
    else:
        print(f"📁 Mode: VIDEO FILE - Source: {source}")
        if not os.path.exists(source):
            print(f"❌ Error: Input file not found -> {source}")
            return
        cap = cv2.VideoCapture(source)

    # 2. Processing Variables
    SKIP_FRAMES = 0
    RESIZE_WIDTH = 640
    
    frame_count = 0
    last_annotated_frame = None

    print(f"✅ Streaming started. Press 'q' to exit.")

    try:
        while True:
            # A. Read Frame
            if is_live:
                if not cap.grabbed: break
                frame = cap.frame.copy()
            else:
                ret, frame = cap.read()
                if not ret: break

            # B. Resize
            if RESIZE_WIDTH:
                h, w = frame.shape[:2]
                scale = RESIZE_WIDTH / w
                frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))

            # C. Detect (every N frames)
            if frame_count % (SKIP_FRAMES + 1) == 0:
                annotated_frame = detect.detect_and_annotate(frame=frame)
                last_annotated_frame = annotated_frame
            else:
                annotated_frame = last_annotated_frame if last_annotated_frame is not None else frame

            # D. Display
            cv2.imshow("Detection Output", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if is_live: cap.stop()
        else: cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use 0 for webcam, or path string for file
    VIDEO_SOURCE = r"assets/t9.mp4" 
    run_system(VIDEO_SOURCE)