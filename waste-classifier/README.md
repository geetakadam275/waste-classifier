# Garbage / Waste Classifier ♻️

Image classification project that identifies the material of a waste item, tells you whether it is **recyclable or non-recyclable**, and gives **disposal guidance**. Built with transfer learning (MobileNetV2) so it can be exported to **TensorFlow Lite** and run on edge devices such as a Raspberry Pi.

## Features
- Image classification into 6 material classes: `cardboard, glass, metal, paper, plastic, trash`
- Recyclable vs non-recyclable decision (mapped from class)
- Disposal guidance per class (`configs/disposal_guide.json`)
- Edge deployment: TFLite export (float16 / int8 quantization) + Raspberry Pi webcam script
- Gradio web demo

## Project structure
```
waste-classifier/
├── app/app.py                  # Gradio web demo
├── configs/disposal_guide.json # class -> recyclable + disposal tips
├── data/README.md              # dataset instructions
├── edge/pi_infer.py            # live camera inference with TFLite (edge)
├── models/                     # saved models (git-ignored)
├── src/
│   ├── config.py               # paths & hyperparameters
│   ├── train.py                # transfer-learning training
│   ├── predict.py              # single-image prediction
│   ├── export_tflite.py        # convert to TFLite
│   └── disposal.py             # recyclable + guidance lookup
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset
Download **TrashNet** (or the Kaggle "Garbage Classification" dataset) and arrange it as described in [`data/README.md`](data/README.md).

## Train
```bash
python src/train.py --epochs 10 --fine_tune_epochs 10
```
Outputs `models/waste_model.keras` and `models/labels.json`.

## Predict
```bash
python src/predict.py path/to/image.jpg
```
Example output:
```
Item: plastic (93.4%)
Recyclable: YES
Disposal: Rinse, remove caps/labels, put in the dry recyclables bin.
```

## Web demo
```bash
python app/app.py
```

## Edge deployment
```bash
python src/export_tflite.py --quantization float16     # or int8, none
# copy models/waste_model.tflite, models/labels.json, configs/ and edge/ to the device
pip install tflite-runtime opencv-python numpy
python edge/pi_infer.py
```

## Notes
- The dataset's `trash` class is treated as non-recyclable; all other classes as recyclable (if clean and dry).
- Recycling rules vary by city/country, so always check local guidelines. Edit `configs/disposal_guide.json` to match yours.

## Ideas to extend
- Add more classes (e-waste, organic/compost, batteries, medical waste)
- Train on a larger dataset such as TACO
- Object detection (YOLO) for multiple items per image
- Use a Pi + servo for an automatic sorting bin

## License
MIT
