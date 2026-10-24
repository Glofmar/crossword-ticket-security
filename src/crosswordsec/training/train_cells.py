"""Train the cell CNN on a folder of labelled images, one sub-folder per label.

    python -m crosswordsec.training.train_cells cells/ --out models/cells.pth

cells/A/*.png, cells/B/*.png, ..., cells/EMPTY/*.png
"""

import argparse
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ..config import load_training
from ..ocr import CellCNN, eval_transform, save_checkpoint, train_transform

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def load_folders(root):
    samples = [(p, p.parent.name) for p in sorted(root.glob("*/*")) if p.suffix.lower() in IMAGE_SUFFIXES]
    return samples, sorted({label for _, label in samples})


def split(samples, val_ratio, seed):
    shuffled = samples[:]
    random.Random(seed).shuffle(shuffled)
    n_val = int(len(shuffled) * val_ratio)
    return shuffled[n_val:], shuffled[:n_val]


class CellDataset(Dataset):
    def __init__(self, samples, index, transform):
        self.samples = samples
        self.index = index
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        return self.transform(Image.open(path).convert("RGB")), self.index[label]


def evaluate(model, loader):
    """Accuracy, per class [correct, total], and (true, predicted) confusion counts."""
    model.eval()
    per_class = defaultdict(lambda: [0, 0])
    confusions = Counter()
    with torch.no_grad():
        for x, y in loader:
            for t, p in zip(y.tolist(), model(x).argmax(1).tolist()):
                per_class[t][0] += t == p
                per_class[t][1] += 1
                if t != p:
                    confusions[(t, p)] += 1
    correct = sum(v[0] for v in per_class.values())
    total = sum(v[1] for v in per_class.values())
    return correct / max(total, 1), per_class, confusions


def train(dataset, out, cfg):
    torch.manual_seed(cfg.seed)
    samples, classes = load_folders(dataset)
    if not samples:
        raise SystemExit(f"no images in {dataset}/<label>/")
    index = {c: i for i, c in enumerate(classes)}
    train_set, val_set = split(samples, cfg.val_ratio, cfg.seed)
    print(f"{len(samples)} images, {len(classes)} classes ({' '.join(classes)}), "
          f"train {len(train_set)} / val {len(val_set)}")

    # some letters are much rarer than others: sample classes evenly
    counts = Counter(label for _, label in train_set)
    sampler = WeightedRandomSampler([1.0 / counts[label] for _, label in train_set], len(train_set))
    augment = train_transform(cfg.image_size, cfg.augment_rotation_deg, cfg.augment_brightness, cfg.augment_contrast)
    train_loader = DataLoader(CellDataset(train_set, index, augment), batch_size=cfg.batch_size, sampler=sampler)
    val_loader = DataLoader(CellDataset(val_set, index, eval_transform(cfg.image_size)), batch_size=cfg.batch_size)

    model = CellCNN(len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg.plateau_patience,
                                                           factor=cfg.plateau_factor)
    loss_fn = nn.CrossEntropyLoss()
    out.parent.mkdir(parents=True, exist_ok=True)

    best, epochs_without_gain = -1.0, 0
    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        model.train()
        loss_sum = correct = total = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        val_acc, _, _ = evaluate(model, val_loader)
        scheduler.step(1 - val_acc)
        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:2d}  loss {loss_sum / total:.4f}  train {correct / total:.1%}  val {val_acc:.1%}  "
              f"lr {lr:.1e}  {time.time() - start:.0f}s", flush=True)
        if val_acc > best:
            best, epochs_without_gain = val_acc, 0
            save_checkpoint(out, model, classes, cfg.image_size)
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= cfg.patience:
                break

    model.load_state_dict(torch.load(out, weights_only=True)["model_state"])
    acc, per_class, confusions = evaluate(model, val_loader)
    print(f"\nvalidation accuracy {acc:.2%} on {len(val_set)} images, saved to {out}")
    weakest = sorted(per_class.items(), key=lambda kv: kv[1][0] / kv[1][1])[:5]
    print("weakest classes:", ", ".join(f"{classes[k]} {c / n:.1%} (n={n})" for k, (c, n) in weakest))
    print("frequent confusions:",
          ", ".join(f"{classes[t]}->{classes[p]} x{c}" for (t, p), c in confusions.most_common(8)) or "none")
    return acc


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the cell classifier.")
    parser.add_argument("dataset", type=Path, help="one sub-folder of images per label")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--training", type=Path, default=Path("configs/training.yaml"))
    args = parser.parse_args(argv)
    train(args.dataset, args.out, load_training(args.training).cells)


if __name__ == "__main__":
    main()
