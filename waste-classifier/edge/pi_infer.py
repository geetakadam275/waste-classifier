"""Live webcam inference with TensorFlow Lite (Raspberry Pi / edge devices).

Needs: models/waste_model.tflite, models/labels.json, configs/disposal_guide.json
Install: pip install tflite-runtime opencv-python numpy
Run:     python edge/pi_infer.py            (press q to quit)
"""
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:  # fall back to full TensorFlow if installed
    from tensorflow.lite.python.interpreter import Interpreter

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "waste_model.tflite"
LABELS = json.loads((ROOT / "models" / "labels.json").read_text())
GUIDE = json.loads((ROOT / "configs" / "disposal_guide.json").read_text())

interpreter = Interpreter(model_path=str(MODEL))
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
H, W = inp["shape"][1], inp["shape"][2]


def classify(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (W, H)).astype(np.float32)
    if inp["dtype"] == np.uint8:
        img = img.astype(np.uint8)
    interpreter.set_tensor(inp["index"], np.expand_dims(img, 0))
    interpreter.invoke()
    probs = interpreter.get_tensor(out["index"])[0].astype(np.float32)
    idx = int(np.argmax(probs))
    if out["dtype"] == np.uint8:
        probs = probs / 255.0
    return LABELS[idx], float(probs[idx])


def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        label, conf = classify(frame)
        info = GUIDE.get(label, {})
        status = "Recyclable" if info.get("recyclable") else "Non-recyclable"
        cv2.putText(frame, f"{label} {conf * 100:.0f}% - {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, info.get("bin", ""), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Waste Classifier", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
