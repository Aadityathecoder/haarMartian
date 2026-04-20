import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

from lane_detection import LineDetector
from martian_detector import MartianDetector


app = Flask(__name__)
CORS(app)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(SCRIPT_DIR, "martian_template.jpg")

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
CAMERA_BACKEND = os.environ.get("CAMERA_BACKEND", "auto").lower()

line_detector = LineDetector()
martian_detector = MartianDetector(TEMPLATE_PATH, threshold=0.55)

latest_frame = None
latest_frame_lock = threading.Lock()

control_state = {
    "up": False,
    "down": False,
    "left": False,
    "right": False,
    "command": "stop",
}

latest_alert = {
    "martian_detected": False,
    "message": "",
    "last_seen": 0.0,
}


class CameraSource:
    def __init__(self):
        self.backend = None
        self.picam2 = None
        self.capture = None
        self.error = ""

        if CAMERA_BACKEND in ("auto", "picamera2"):
            self._try_picamera2()

        if self.backend is None and CAMERA_BACKEND in ("auto", "opencv"):
            self._try_opencv()

    def _try_picamera2(self):
        try:
            from picamera2 import Picamera2

            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(1)
            self.backend = "picamera2"
            self.error = ""
            print("Camera backend: picamera2")
        except Exception as exc:
            self.picam2 = None
            self.error = f"picamera2 failed: {exc}"
            print(self.error)

    def _try_opencv(self):
        self.capture = cv2.VideoCapture(CAMERA_INDEX)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if self.capture.isOpened():
            self.backend = "opencv"
            self.error = ""
            print(f"Camera backend: OpenCV VideoCapture({CAMERA_INDEX})")
        else:
            self.capture.release()
            self.capture = None
            self.error = f"OpenCV camera {CAMERA_INDEX} could not be opened"
            print(self.error)

    def is_opened(self):
        return self.backend is not None

    def read(self):
        if self.backend == "picamera2":
            rgb = self.picam2.capture_array()
            return True, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if self.backend == "opencv":
            return self.capture.read()

        return False, None


camera = CameraSource()


def make_placeholder_frame(message):
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.putText(
        frame,
        message,
        (30, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        frame,
        "Check camera connection/backend.",
        (30, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
    )
    return frame


def camera_reader():
    global latest_frame

    while True:
        ok, frame = camera.read()

        if ok and frame is not None:
            with latest_frame_lock:
                latest_frame = frame.copy()
            time.sleep(0.01)
        else:
            with latest_frame_lock:
                latest_frame = make_placeholder_frame(camera.error or "Camera not available")
            time.sleep(0.5)


def get_latest_frame():
    with latest_frame_lock:
        if latest_frame is None:
            return None
        return latest_frame.copy()


def encode_frame(frame):
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return buffer.tobytes()


def mjpeg_chunk(frame):
    jpg = encode_frame(frame)
    if jpg is None:
        return None

    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"


def generate_raw_frames():
    while True:
        frame = get_latest_frame()

        if frame is None:
            frame = make_placeholder_frame("Waiting for camera")

        chunk = mjpeg_chunk(frame)
        if chunk is not None:
            yield chunk

        time.sleep(0.03)


def generate_processed_frames():
    global latest_alert

    while True:
        frame = get_latest_frame()

        if frame is None:
            frame = make_placeholder_frame("Waiting for camera")

        try:
            processed = line_detector.process_frame(frame)
        except Exception:
            processed = frame.copy()

        found, boxes = martian_detector.detect(frame)

        if found:
            latest_alert = {
                "martian_detected": True,
                "message": "MARTIAN DETECTED!!!",
                "last_seen": time.time(),
            }
            processed = martian_detector.draw_detections(processed, boxes)

        chunk = mjpeg_chunk(processed)
        if chunk is not None:
            yield chunk

        time.sleep(0.03)


@app.route("/")
def home():
    return jsonify(
        {
            "ok": True,
            "message": "Robot vision server is running",
            "camera_backend": camera.backend,
            "camera_error": camera.error,
        }
    )


@app.route("/gui")
def gui():
    return send_from_directory(SCRIPT_DIR, "gui.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_raw_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/video_feed/processed")
def processed_video_feed():
    return Response(
        generate_processed_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/alerts/status")
def alerts_status():
    age = time.time() - latest_alert["last_seen"]
    active = latest_alert["martian_detected"] and age <= 3

    return jsonify(
        {
            "martian_detected": active,
            "message": latest_alert["message"] if active else "",
        }
    )


@app.route("/control/set", methods=["POST"])
def control_set():
    global control_state

    data = request.get_json(force=True, silent=True)

    if data is None:
        return jsonify({"ok": False, "error": "Invalid JSON"}), 400

    control_state = {
        "up": bool(data.get("up", False)),
        "down": bool(data.get("down", False)),
        "left": bool(data.get("left", False)),
        "right": bool(data.get("right", False)),
        "command": data.get("command", "stop"),
    }

    print(f"CONTROL STATE: {control_state}")

    return jsonify({"ok": True, "state": control_state})


@app.route("/control/state")
def control_get():
    return jsonify({"ok": True, "state": control_state})


if __name__ == "__main__":
    reader_thread = threading.Thread(target=camera_reader, daemon=True)
    reader_thread.start()

    app.run(host=HOST, port=PORT, threaded=True)
