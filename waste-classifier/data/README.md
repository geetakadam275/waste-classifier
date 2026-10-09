# Dataset

This project expects an image-folder dataset split into `train/` and `val/`, one sub-folder per class:

```
data/
├── train/
│   ├── cardboard/
│   ├── glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   └── trash/
└── val/
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

## Where to get data
- **TrashNet** (~2,500 images, 6 classes): https://github.com/garythung/trashnet
- **Garbage Classification** on Kaggle (6 classes, similar to TrashNet)
- **TACO** (more realistic, in-the-wild litter): http://tacodataset.org

## Splitting (80/20)
After downloading all images into `data/raw/<class>/`, run:

```bash
python - <<'EOF'
import random, shutil
from pathlib import Path
random.seed(42)
raw = Path("data/raw")
for cls in raw.iterdir():
    imgs = list(cls.glob("*.*")); random.shuffle(imgs)
    cut = int(0.8 * len(imgs))
    for split, items in (("train", imgs[:cut]), ("val", imgs[cut:])):
        out = Path("data") / split / cls.name; out.mkdir(parents=True, exist_ok=True)
        for p in items: shutil.copy(p, out / p.name)
EOF
```

Image files are git-ignored; only this README is tracked.
