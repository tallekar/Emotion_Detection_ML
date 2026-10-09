"""
Emotion Detector — OpenCV + DeepFace
Live webcam emotion detection with a styled HUD overlay.

Run: python emotion_detector.py
Press Q to quit, S to save screenshot, P to pause/resume.
"""

import cv2
import numpy as np
from deepface import DeepFace
import threading
import time
import os
from datetime import datetime
from collections import deque

# ─── Config ───────────────────────────────────────────────────────────────────
WINDOW_NAME  = "Emotion Detector — Press Q to quit"
ANALYZE_FPS  = 5          # DeepFace analysis rate (heavy, runs in thread)
SMOOTH_N     = 6          # Frames to smooth emotion bars over
BACKEND      = "mtcnn"    # Face detector: opencv / retinaface / mtcnn
SCREENSHOT_DIR = "screenshots"

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

EMOTION_COLORS_BGR = {
    "angry":    (60,  60,  226),
    "disgust":  (60,  180, 60),
    "fear":     (180, 120, 40),
    "happy":    (50,  200, 50),
    "sad":      (200, 100, 50),
    "surprise": (200, 60,  180),
    "neutral":  (160, 160, 160),
}

EMOTION_EMOJIS = {
    "angry":    "ANGRY",
    "disgust":  "DISGUST",
    "fear":     "FEAR",
    "happy":    "HAPPY",
    "sad":      "SAD",
    "surprise": "SURPRISE",
    "neutral":  "NEUTRAL",
}


# ─── State ────────────────────────────────────────────────────────────────────
class DetectorState:
    def __init__(self):
        self.lock = threading.Lock()
        self.emotions: dict  = {e: 0.0 for e in EMOTIONS}
        self.dominant: str   = "neutral"
        self.face_box: tuple = None   # (x,y,w,h)
        self.age: int        = None
        self.gender: str     = None
        self.fps_cam: float  = 0.0
        self.fps_ai:  float  = 0.0
        self.paused: bool    = False
        self.history         = {e: deque([0.0]*SMOOTH_N, maxlen=SMOOTH_N) for e in EMOTIONS}
        self.smoothed        = {e: 0.0 for e in EMOTIONS}
        self.no_face: bool   = True
        self.frame_count: int = 0


state = DetectorState()


# ─── DeepFace analysis thread ─────────────────────────────────────────────────
def analysis_worker(frame_queue):
    interval = 1.0 / ANALYZE_FPS
    ai_times = deque(maxlen=10)

    while True:
        if frame_queue:
            frame = frame_queue[-1]   # always use latest
        else:
            time.sleep(0.05)
            continue

        t0 = time.time()
        try:
            # Auto-contrast fix for backlit / bright background conditions
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            frame_small = cv2.resize(frame, (640, 480))

            results = DeepFace.analyze(
                frame_small,
                actions=["emotion", "age", "gender"],
                enforce_detection=True,
                detector_backend=BACKEND,
                silent=True,
            )
            r = results[0] if isinstance(results, list) else results
            emo_raw = r.get("emotion", {})
            total   = sum(emo_raw.values()) or 1
            emo_norm = {e: emo_raw.get(e, 0.0) / total * 100 for e in EMOTIONS}
            dominant = max(emo_norm, key=emo_norm.get)
            region   = r.get("region", {})
            if region and region.get("w",0)>0:
                   h_ratio = frame.shape[0] / 480
                   w_ratio = frame.shape[1] / 640
                   box = (
                        int(region.get("x", 0) * w_ratio),
                        int(region.get("y", 0) * h_ratio),
                        int(region.get("w", 0) * w_ratio),
                        int(region.get("h", 0) * h_ratio),
                    )
            else:
                box=None
            print("Region:", r.get("region"))

            with state.lock:
                state.emotions  = emo_norm
                state.dominant  = dominant
                state.face_box  = box
                state.age       = r.get("age")
                state.gender    = r.get("dominant_gender", "")
                state.no_face   = False
                for e in EMOTIONS:
                    state.history[e].append(emo_norm[e])
                    state.smoothed[e] = sum(state.history[e]) / SMOOTH_N

        except Exception:
            with state.lock:
                state.no_face  = True
                state.face_box = None
                for e in EMOTIONS:
                    state.history[e].append(0.0)
                    state.smoothed[e] = sum(state.history[e]) / SMOOTH_N

        elapsed = time.time() - t0
        ai_times.append(1.0 / max(elapsed, 0.001))
        with state.lock:
            state.fps_ai = sum(ai_times) / len(ai_times)

        sleep_t = max(0, interval - elapsed)
        time.sleep(sleep_t)


# ─── HUD drawing helpers ───────────────────────────────────────────────────────
def draw_rounded_rect(img, x, y, w, h, r, color, thickness=-1, alpha=1.0):
    overlay = img.copy()
    cv2.rectangle(overlay, (x+r, y), (x+w-r, y+h), color, thickness)
    cv2.rectangle(overlay, (x, y+r), (x+w, y+h-r), color, thickness)
    for cx, cy in [(x+r, y+r),(x+w-r, y+r),(x+r, y+h-r),(x+w-r, y+h-r)]:
        cv2.circle(overlay, (cx, cy), r, color, thickness)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
    else:
        img[:] = overlay


def draw_panel(img, x, y, w, h, alpha=0.55):
    sub = img[y:y+h, x:x+w]
    black = np.zeros_like(sub)
    cv2.addWeighted(black, alpha, sub, 1-alpha, 0, sub)
    img[y:y+h, x:x+w] = sub
    cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,255,30), 1)


def put_text(img, text, x, y, scale=0.5, color=(255,255,255), thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, text, (x, y), font, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color,   thickness,   cv2.LINE_AA)


def draw_emotion_bars(img, x, y, w):
    bar_h   = 14
    gap     = 22
    label_w = 80

    with state.lock:
        smoothed  = dict(state.smoothed)
        dominant  = state.dominant

    for i, emo in enumerate(EMOTIONS):
        val   = smoothed[emo]
        bx    = x + label_w
        by    = y + i * gap
        bw    = w - label_w - 10
        fill  = int(bw * val / 100)
        color = EMOTION_COLORS_BGR[emo]
        is_dom = (emo == dominant)

        # bar bg
        cv2.rectangle(img, (bx, by), (bx+bw, by+bar_h), (60,60,60), -1)
        # bar fill
        if fill > 0:
            cv2.rectangle(img, (bx, by), (bx+fill, by+bar_h), color, -1)
        # highlight dominant
        if is_dom:
            cv2.rectangle(img, (bx, by), (bx+bw, by+bar_h), color, 1)

        label_color = color if is_dom else (180,180,180)
        put_text(img, emo.upper()[:7], x, by+11, 0.36, label_color, 1)
        put_text(img, f"{val:.0f}%", bx+bw+4, by+11, 0.36, (200,200,200), 1)


def draw_face_box(img):
    with state.lock:
        box    = state.face_box
        dom    = state.dominant
        no_face = state.no_face

    if no_face or box is None:
        return

    x, y, w, h = box
    color  = EMOTION_COLORS_BGR.get(dom, (200,200,200))
    corner = 16

    # Draw corner brackets
    for (sx, sy, dx, dy) in [
        (x,   y,   1,  1), (x+w, y,   -1, 1),
        (x,   y+h, 1, -1), (x+w, y+h, -1,-1),
    ]:
        cv2.line(img, (sx, sy), (sx + dx*corner, sy), color, 2)
        cv2.line(img, (sx, sy), (sx, sy + dy*corner), color, 2)

    label = EMOTION_EMOJIS.get(dom, dom.upper())
    (text_size, _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    tw = text_size[0]
    lx = x + w//2 - tw//2
    put_text(img, label, lx, y-8, 0.5, color, 1)
  
   


def draw_hud(img):
    H, W = img.shape[:2]
    panel_w = 220
    panel_h = SMOOTH_N * 0 + len(EMOTIONS) * 22 + 70
    px, py  = 10, 10

    draw_panel(img, px, py, panel_w, panel_h)

    with state.lock:
        dom    = state.dominant
        fps_c  = state.fps_cam
        fps_a  = state.fps_ai
        age    = state.age
        gender = state.gender
        no_face = state.no_face
        paused  = state.paused

    # Title
    put_text(img, "EMOTION DETECTOR", px+8, py+18, 0.46, (220,220,220), 1)
    put_text(img, f"CAM {fps_c:.0f}fps  AI {fps_a:.1f}fps", px+8, py+34, 0.35, (140,140,140), 1)

    if not no_face:
        dom_color = EMOTION_COLORS_BGR.get(dom, (200,200,200))
        put_text(img, dom.upper(), px+8, py+52, 0.55, dom_color, 2)
        info = []
        if age:   info.append(f"Age ~{age}")
        if gender: info.append(gender.title())
        if info:
            put_text(img, "  ".join(info), px+8, py+66, 0.36, (160,160,160), 1)
        draw_emotion_bars(img, px+6, py+76, panel_w-12)
    else:
        put_text(img, "NO FACE DETECTED", px+8, py+60, 0.42, (100,100,220), 1)

    # Bottom hints
    hints = "Q quit  S screenshot  P pause"
    put_text(img, hints, 8, H-10, 0.35, (120,120,120), 1)

    if paused:
        put_text(img, "PAUSED", W//2-40, H//2, 0.9, (80,80,255), 2)


# ─── Main loop ────────────────────────────────────────────────────────────────
def main():
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_queue = []
    t = threading.Thread(target=analysis_worker, args=(frame_queue,), daemon=True)
    t.start()

    cam_times = deque(maxlen=30)
    prev_t    = time.time()

    print("[INFO] Starting — press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        cam_times.append(1.0 / max(now - prev_t, 0.001))
        prev_t = now

        with state.lock:
            state.fps_cam    = sum(cam_times) / len(cam_times)
            state.frame_count += 1
            paused = state.paused

        if not paused:
            frame_queue.clear()
            frame_queue.append(frame.copy())

        draw_face_box(frame)
        draw_hud(frame)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            with state.lock:
                state.paused = not state.paused
        elif key == ord('s'):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SCREENSHOT_DIR, f"emotion_{ts}.png")
            cv2.imwrite(path, frame)
            print(f"[INFO] Screenshot saved: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
