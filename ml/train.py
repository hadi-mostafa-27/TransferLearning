import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model_def import build_model, freeze_backbone, unfreeze_last_blocks_resnet
from utils import evaluate


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def assert_dataset_structure(data_dir: Path):
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing folder: {split_dir}")

        classes = [d.name for d in split_dir.iterdir() if d.is_dir()]
        if len(classes) < 2:
            raise RuntimeError(
                f"{split_dir} must contain at least 2 class folders. Found: {classes}"
            )


def main():
    parser = argparse.ArgumentParser("Transfer Learning Trainer")
    parser.add_argument("--data_dir", type=str, required=True, help="Path with train/val/test")
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs_head", type=int, default=3)
    parser.add_argument("--epochs_ft", type=int, default=3)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_ft", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assert_dataset_structure(data_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    tfm_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    tfm_eval = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_ds = datasets.ImageFolder(data_dir / "train", transform=tfm_train)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=tfm_eval)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=tfm_eval)

    print(f"[INFO] Classes: {train_ds.classes}")
    print(f"[INFO] Train samples: {len(train_ds)}")
    print(f"[INFO] Val samples:   {len(val_ds)}")
    print(f"[INFO] Test samples:  {len(test_ds)}")

    class_to_idx = train_ds.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = build_model(
        num_classes=len(train_ds.classes),
        backbone=args.backbone,
        pretrained=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # =========================
    # Phase 1 — Train head only
    # =========================
    freeze_backbone(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_head,
    )

    print("\n[PHASE 1] Training classifier head")
    for epoch in range(args.epochs_head):
        model.train()
        pbar = tqdm(train_loader, desc=f"Head Epoch {epoch+1}/{args.epochs_head}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_metrics = evaluate(model, val_loader, device)
        print(f"[VAL][HEAD] AUROC: {val_metrics['auroc']}")

    # =========================
    # Phase 2 — Fine-tuning
    # =========================
    unfreeze_last_blocks_resnet(model, n_blocks=1)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_ft,
    )

    print("\n[PHASE 2] Fine-tuning last ResNet block")
    for epoch in range(args.epochs_ft):
        model.train()
        pbar = tqdm(train_loader, desc=f"FT Epoch {epoch+1}/{args.epochs_ft}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_metrics = evaluate(model, val_loader, device)
        print(f"[VAL][FT] AUROC: {val_metrics['auroc']}")

    # =========================
    # Final test evaluation
    # =========================
    test_metrics = evaluate(model, test_loader, device)
    print(f"\n[TEST] AUROC: {test_metrics['auroc']}")

    # =========================
    # Save artifacts
    # =========================
    torch.save(model.state_dict(), out_dir / "model.pt")

    meta = {
        "backbone": args.backbone,
        "img_size": args.img_size,
        "classes": idx_to_class,
        "class_to_idx": class_to_idx,
        "metrics_test": test_metrics,
        "seed": args.seed,
        "device_trained_on": device,
    }

    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[SAVED] {out_dir / 'model.pt'}")
    print(f"[SAVED] {out_dir / 'model_meta.json'}")


if __name__ == "__main__":
    main()
