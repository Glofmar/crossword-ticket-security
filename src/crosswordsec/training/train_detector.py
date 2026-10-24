"""Train the YOLOv8-OBB detector (draw panel and grid) on an annotated dataset.

    python -m crosswordsec.training.train_detector annotations/ --out models/detector_obb.pt

The dataset is in YOLOv8 oriented-box format: a data.yaml with
the class names and train/, valid/, test/ folders, each with images/ and labels/.
"""

import argparse
import shutil
from pathlib import Path

import yaml

from ..config import load_training


def dataset_yaml(dataset, runs):
    # rewrite data.yaml with an absolute path, otherwise ultralytics looks for the
    # images relative to its own datasets folder
    exported = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    names = exported["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    data = {"path": str(dataset.resolve()), "train": "train/images", "val": "valid/images",
            "test": "test/images", "names": dict(enumerate(names))}
    runs.mkdir(parents=True, exist_ok=True)
    path = runs / "data.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def train(dataset, out, runs, cfg):
    from ultralytics import YOLO

    data = dataset_yaml(dataset, runs)
    YOLO(cfg.base_model).train(data=str(data), epochs=cfg.epochs, imgsz=cfg.image_size, batch=cfg.batch,
                               device=cfg.device, workers=cfg.workers, seed=cfg.seed, deterministic=True,
                               project=str(runs.resolve()), name="detector", exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(runs / "detector" / "weights" / "best.pt", out)

    metrics = YOLO(str(out)).val(data=str(data), split="test", imgsz=cfg.image_size, batch=cfg.batch,
                                 device=cfg.device, workers=cfg.workers, plots=False, verbose=False,
                                 project=str(runs.resolve()), name="test", exist_ok=True)
    print(f"\ntest mAP50 {metrics.box.map50:.3f}, mAP50-95 {metrics.box.map:.3f}, saved to {out}")
    for i, name in metrics.names.items():
        if i < len(metrics.box.maps):
            print(f"  {name}: mAP50-95 {metrics.box.maps[i]:.3f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the draw panel / grid detector.")
    parser.add_argument("dataset", type=Path, help="YOLOv8-OBB export")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="ultralytics logs and checkpoints")
    parser.add_argument("--training", type=Path, default=Path("configs/training.yaml"))
    args = parser.parse_args(argv)
    train(args.dataset, args.out, args.runs, load_training(args.training).detector)


if __name__ == "__main__":
    main()
