from ultralytics import YOLO
import cv2
import subprocess
import shutil
import os

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

air_model = YOLO("models/air_model.pt")
ground_model = YOLO("models/ground_model_v2.pt")


def get_model(mode):
    if mode == "Air Surveillance":
        return air_model
    else:
        return ground_model


# --------------------------------------------------
# GLOBAL ZONE (updated from Streamlit)
# --------------------------------------------------

ZONE = None


def set_zone(x1, y1, x2, y2):
    global ZONE
    ZONE = (x1, y1, x2, y2)


def get_zone(frame):
    global ZONE
    h, w = frame.shape[:2]

    if ZONE is None:
        return int(w * 0.3), int(h * 0.3), int(w * 0.7), int(h * 0.7)

    return ZONE


# --------------------------------------------------
# PROCESS IMAGE
# --------------------------------------------------

def process_image(frame, mode="GROUND", conf=0.25):

    model = get_model(mode)
    results = model.predict(frame, conf=conf)
    boxes = results[0].boxes

    x1, y1, x2, y2 = get_zone(frame)

    detections = []

    # Draw restricted zone
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(
        frame,
        "RESTRICTED ZONE",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    for box in boxes:

        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf_score = float(box.conf[0])
        label = model.names[cls]

        cx = (bx1 + bx2) // 2
        cy = (by1 + by2) // 2

        inside = (x1 <= cx <= x2 and y1 <= cy <= y2)

        # Threat logic
        if inside and conf_score > 0.6:
            threat = "HIGH RISK"
            color = (0, 0, 255)
        elif inside:
            threat = "MEDIUM"
            color = (0, 165, 255)
        else:
            threat = "LOW"
            color = (0, 255, 0)

        detections.append({"label": label, "conf": conf_score, "threat": threat})

        # Draw bounding box
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 3)

        text = f"{label} {conf_score:.2f} | {threat}"

        (label_w, label_h), _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        # Draw label background
        cv2.rectangle(
            frame,
            (bx1, by1 - label_h - 10),
            (bx1 + label_w + 5, by1),
            (0, 0, 0),
            -1
        )

        # Draw label text
        cv2.putText(
            frame,
            text,
            (bx1, by1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    return frame, detections


# --------------------------------------------------
# PROCESS VIDEO
# --------------------------------------------------

def process_video(path, mode="GROUND", conf=0.25):

    cap = cv2.VideoCapture(path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 25

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_path   = "temp_processed.mp4"
    output_path = "processed_video.mp4"

    # Process every 5th frame → effective FPS = original_fps / 5
    output_fps = max(fps / 5, 1)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, output_fps, (width, height))

    frame_count   = 0
    frame_written = 0
    total_objects = 0
    total_threats = 0
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip non-keyframes (process every 5th frame only)
        if frame_count % 5 != 0:
            continue

        frame, detections = process_image(frame, mode=mode, conf=conf)
        # total_objects += len(detections)
        # total_threats += sum(1 for d in detections if d["threat"] == "HIGH RISK")
        total_objects += len(detections)

        total_threats += sum(
            1 for d in detections
            if d["threat"] == "HIGH RISK"
        )
        out.write(frame)
        frame_written += 1

    cap.release()
    out.release()

    print(f"Frames read = {frame_count}")
    print(f"Frames written = {frame_written}")


    ffmpeg_available = shutil.which("ffmpeg") is not None

    if ffmpeg_available:
        print("ffmpeg found – re-encoding to H.264 ...")
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", temp_path,
                "-vcodec", "libx264",
                "-crf", "23",
                "-preset", "fast",
                "-movflags", "+faststart",
                output_path
            ],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"Output size = {os.path.getsize(output_path)} bytes")
            return output_path, total_objects, total_threats
        else:
            print("ffmpeg re-encode failed:", result.stderr)
            # fall through to avc1 attempt

    # --- Attempt 2: re-write with avc1 codec via OpenCV ---
    print("Trying avc1 codec via OpenCV ...")
    fourcc_avc1 = cv2.VideoWriter_fourcc(*'avc1')
    out2 = cv2.VideoWriter(output_path, fourcc_avc1, output_fps, (width, height))

    if out2.isOpened():
        cap2 = cv2.VideoCapture(temp_path)
        while True:
            ret2, frm2 = cap2.read()
            if not ret2:
                break
            out2.write(frm2)
        cap2.release()
        out2.release()

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"avc1 output size = {os.path.getsize(output_path)} bytes")
            return output_path, total_objects, total_threats
        else:
            print("avc1 produced empty file, falling back to mp4v")
    else:
        print("avc1 codec not available, falling back to mp4v")

    # --- Last resort: return the mp4v file ---
    print("Returning raw mp4v file (may not play in browser)")
    return temp_path, total_objects, total_threats
