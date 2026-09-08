#!/usr/bin/env python
"""Step 4 (free path): train a court-crop classifier on CPU.

Fine-tunes a small pretrained CNN (torchvision ResNet-18) to classify 160 x 320
court crops into the five classes. A few hundred labeled crops are enough to
start; CPU training on a GitHub runner takes minutes, not hours.

Input:  data/crops/train/<class>/*.jpg   (from 03_make_crops.py --export)
Output: data/models/classifier.pt         (weights + class list + input size)
        data/models/classifier_report.json (per-class precision/recall on the val split)

Usage::

    python scripts/04_train_classifier.py
    python scripts/04_train_classifier.py --epochs 25 --arch resnet34
    python scripts/04_train_classifier.py --val-frac 0.25 --seed 1

Val split is by site (hash of site_id), so a court never appears in both
splits. Classes with no examples are still in the output layer but will
simply never be predicted with confidence; the report says which.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, DATA, MODELS_DIR, log, setup_logging  # noqa: E402

CROPS_DIR = DATA / "crops"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def site_of(path: Path) -> str:
    return path.stem.split("__")[0]


def build_model(arch: str, n_classes: int, pretrained: bool = False):
    """ResNet with a fresh 5-way head. ``pretrained`` loads ImageNet weights
    (needs network on first use); on any download failure it falls back to
    random init with a warning rather than aborting the run."""
    import torch.nn as nn
    import torchvision

    ctor = torchvision.models.resnet34 if arch == "resnet34" else torchvision.models.resnet18
    weights = None
    if pretrained:
        weights = (torchvision.models.ResNet34_Weights if arch == "resnet34" else torchvision.models.ResNet18_Weights).IMAGENET1K_V1
    try:
        m = ctor(weights=weights)
    except Exception as e:  # download blocked / offline
        log.warning("could not load pretrained weights (%s); training from scratch", e)
        m = ctor(weights=None)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops-dir", type=Path, default=CROPS_DIR / "train")
    ap.add_argument("--out", type=Path, default=MODELS_DIR / "classifier.pt")
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet34"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-pretrained", action="store_true", help="random init instead of ImageNet weights (offline tests)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    try:
        import torch
        import torch.nn as nn
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
    except ImportError:
        log.error("torch/torchvision missing: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        return 2

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    files = [(p, p.parent.name) for p in sorted(args.crops_dir.glob("*/*.jpg")) if p.parent.name in CLASSES]
    if len(files) < 20:
        log.error("only %d labeled crops in %s; label more sheets first", len(files), args.crops_dir)
        return 2
    val = [(p, c) for p, c in files if int(hashlib.sha1(site_of(p).encode()).hexdigest(), 16) % 1000 < args.val_frac * 1000]
    train = [f for f in files if f not in val]
    log.info("%d train / %d val crops; class counts %s", len(train), len(val), dict(Counter(c for _, c in files)))

    norm = transforms.Normalize(MEAN, STD)
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.02),
        transforms.RandomAffine(degrees=4, translate=(0.04, 0.04), scale=(0.92, 1.08)),
        transforms.ToTensor(), norm])
    val_tf = transforms.Compose([transforms.ToTensor(), norm])

    class Crops(Dataset):
        def __init__(self, items, tf):
            self.items, self.tf = items, tf
        def __len__(self):
            return len(self.items)
        def __getitem__(self, i):
            p, c = self.items[i]
            with Image.open(p) as im:
                return self.tf(im.convert("RGB")), CLASSES.index(c)

    # class-balanced sampling weights so rare classes are not ignored
    counts = Counter(c for _, c in train)
    weights = [1.0 / counts[c] for _, c in train]
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(train), replacement=True)
    dl_train = DataLoader(Crops(train, train_tf), batch_size=args.batch, sampler=sampler, num_workers=args.workers)
    dl_val = DataLoader(Crops(val, val_tf), batch_size=args.batch, num_workers=args.workers) if val else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.arch, len(CLASSES), pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_acc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = n = 0
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y); n += len(y)
        sched.step()
        acc, report = evaluate(model, dl_val, device) if dl_val else (float("nan"), {})
        log.info("epoch %d/%d loss %.3f val_acc %.3f", epoch, args.epochs, tot / max(n, 1), acc)
        if not dl_val or acc >= best_acc:
            best_acc, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_report = report

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": args.arch, "classes": CLASSES, "state_dict": best_state, "input_size": [160, 320],
                "mean": MEAN, "std": STD, "val_acc": best_acc, "n_train": len(train), "n_val": len(val)}, args.out)
    (args.out.parent / "classifier_report.json").write_text(json.dumps(
        {"val_acc": best_acc, "n_train": len(train), "n_val": len(val), "class_counts": dict(Counter(c for _, c in files)),
         "per_class": best_report}, indent=2))
    log.info("saved %s (val acc %.3f); report %s", args.out, best_acc, args.out.parent / "classifier_report.json")
    return 0


def evaluate(model, dl, device) -> tuple[float, dict]:
    import torch
    model.eval()
    correct = total = 0
    tp = Counter(); fp = Counter(); fn = Counter()
    with torch.no_grad():
        for x, y in dl:
            pred = model(x.to(device)).argmax(1).cpu()
            for p, t in zip(pred.tolist(), y.tolist()):
                total += 1
                if p == t:
                    correct += 1; tp[t] += 1
                else:
                    fp[p] += 1; fn[t] += 1
    report = {}
    for i, c in enumerate(CLASSES):
        n = tp[i] + fn[i]
        report[c] = {"support": n,
                     "precision": round(tp[i] / (tp[i] + fp[i]), 3) if tp[i] + fp[i] else None,
                     "recall": round(tp[i] / n, 3) if n else None}
    return (correct / total if total else float("nan")), report


if __name__ == "__main__":
    sys.exit(main())
