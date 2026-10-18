# -*- coding: utf-8 -*-
"""Binary classification of F0 and F1 crystal images with external validation.

Training data:
- F0: images listed in the first column of the F0 guide spreadsheet.
- F1: rows with stage value 1 in the third column of the fibrosis guide.

External evaluation data:
- F0: all images in the designated F0 image directory.
- F1: rows with stage value 1 in the third column of the fibrosis guide.

Outputs include per-epoch sample predictions, evaluation metrics, model checkpoints,
and training logs. This script preserves the original training and evaluation workflow.
"""

import os, math, random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, ConcatDataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

import timm
from tqdm import tqdm

from pathlib import Path

# Project-relative paths; override these with environment variables when needed.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("CRYSTANET_DATA_ROOT", str(PROJECT_ROOT / "data")))
PRETRAINED_WEIGHTS = Path(
    os.environ.get(
        "CRYSTANET_PRETRAINED_WEIGHTS",
        str(PROJECT_ROOT / "weights" / "convnext_base.fb_in22k_ft_in1k" / "pytorch_model.bin"),
    )
)


# Configuration
CONFIG = {
    # Internal training data
    "TRAIN_F1_DIR":   DATA_ROOT / "internal" / "f1_images",
    "TRAIN_F1_XLSX":  DATA_ROOT / "guides" / "liver_fibrosis_list.xlsx",
    "TRAIN_F0_DIR":   DATA_ROOT / "internal" / "f0_images",
    "TRAIN_F0_XLSX":  DATA_ROOT / "guides" / "f0_image_list.xlsx",

    # External evaluation data from Zengcheng
    "TEST_F1_DIR":    DATA_ROOT / "zengcheng" / "f1_images",
    "TEST_F1_XLSX":   DATA_ROOT / "guides" / "zengcheng_liver_fibrosis_list.xlsx",
    "TEST_F0_DIR":    DATA_ROOT / "zengcheng" / "f0_images",

    # Training parameters
    "model_name": "convnext_base",
    "checkpoint_path": str(PRETRAINED_WEIGHTS),  # Set CRYSTANET_PRETRAINED_WEIGHTS to the checkpoint path.
    "num_classes": 2,
    "epochs": 20,
    "batch_size": 16,
    "num_workers": 8,             # Set to 0 if multiprocessing causes issues on Windows.
    "lr": 1e-7,
    "weight_decay": 1e-2,
    "step_size": 3,
    "gamma": 0.5,
    "img_size": 128,
    "seed": 1024,
    "dropout": 0.7,

    # Class-balancing options
    "class_weight_mode": "manual",     # "auto" | "manual" | None
    "manual_class_weights": [  ,  ],   # Used when mode="manual"; order is [F0 weight, F1 weight].

    "use_focal_loss": False,         # True selects FocalLoss.
    "focal_gamma": 2.0,
    "focal_alpha": None,             # For example, [0.7, 1.6]. If None, reuse the cross-entropy class weights.

    "use_weighted_sampler": False,   # Enable WeightedRandomSampler when True.

    # Output directories
    "out_data_dir": "epoch-data",
    "out_model_dir": "epoch-models",
    "log_dir": "epoch-logs",
}

# Utility functions
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def log(msg: str):
    print(msg)
    with open(Path(CONFIG["log_dir"]) / "training_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}\n")

ALLOWED_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

def normalize_stage_to_01(v):
    s = str(v).strip().lower()
    # Keep compatibility with legacy spreadsheets that use Chinese stage labels.
    if s in ["1","1.0","f1","1级","stage1","stage 1","Ⅰ","i","grade1","grade 1"]:
        return 1
    if s in ["0","0.0","f0","0级","stage0","stage 0"]:
        return 0
    try:
        fv = float(s)
        if abs(fv - 1.0) < 1e-9: return 1
        if abs(fv - 0.0) < 1e-9: return 0
    except:
        pass
    return None

# Dataset definitions
class ExcelListDataset(Dataset):
    """
    For F0 training, read the filename allowlist from the first Excel column and match files in the image directory. Files may have missing or alternative extensions. All matched images receive label 0.
    """
    def __init__(self, xlsx_path, img_dir, label=0, transform=None):
        self.transform = transform
        self.img_label = int(label)
        self.records = []

        df = pd.read_excel(xlsx_path)
        if df.shape[1] < 1:
            log(f"[WARN] {xlsx_path} The spreadsheet has no first column; returning an empty dataset.")
            return
        names = [str(x).strip() for x in df.iloc[:, 0].tolist()]

        img_dir = Path(img_dir)
        path_by_name, path_by_stem = {}, {}
        for p in img_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
                path_by_name[p.name.lower()] = str(p)
                path_by_stem[p.stem.lower()] = str(p)

        add_cnt, miss_cnt = 0, 0
        for nm in names:
            key = nm.lower()
            cand = None
            if Path(key).suffix:
                cand = path_by_name.get(key)
            if cand is None:
                cand = path_by_stem.get(Path(key).stem.lower())
            if cand is not None and os.path.exists(cand):
                self.records.append((cand, self.img_label, os.path.basename(cand)))
                add_cnt += 1
            else:
                miss_cnt += 1

        log(f"[ExcelListDataset-F0] xlsx={Path(xlsx_path).name}, dir={img_dir}, included={add_cnt}, missing={miss_cnt}")

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        img_path, label, filename = self.records[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, label, filename


class ExcelF1Dataset(Dataset):
    """
    For F1 training/evaluation, retain rows with stage value 1 in the third Excel column, match filenames from the first column, and assign label 1.
    """
    def __init__(self, xlsx_path, img_dir, transform=None):
        self.transform = transform
        self.img_label = 1
        self.records = []

        df = pd.read_excel(xlsx_path)
        if df.shape[1] < 3:
            log(f"[WARN] {xlsx_path} The spreadsheet has no third column; the F1 dataset will be empty.")
            return

        col3 = df.iloc[:, 2].map(normalize_stage_to_01)
        df = df[pd.Series(col3).eq(1).values]
        names = [str(x).strip() for x in df.iloc[:, 0].tolist()]

        img_dir = Path(img_dir)
        path_by_name, path_by_stem = {}, {}
        for p in img_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
                path_by_name[p.name.lower()] = str(p)
                path_by_stem[p.stem.lower()] = str(p)

        add_cnt, miss_cnt = 0, 0
        for nm in names:
            key = nm.lower()
            cand = None
            if Path(key).suffix:
                cand = path_by_name.get(key)
            if cand is None:
                cand = path_by_stem.get(Path(key).stem.lower())
            if cand is not None and os.path.exists(cand):
                self.records.append((cand, self.img_label, os.path.basename(cand)))
                add_cnt += 1
            else:
                miss_cnt += 1

        log(f"[ExcelF1Dataset] xlsx={Path(xlsx_path).name}, dir={img_dir}, F1included={add_cnt}, missing={miss_cnt}")

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        img_path, label, filename = self.records[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, label, filename


class DirectoryDataset(Dataset):
    """For external F0 evaluation, include every supported image in the directory and assign label 0."""
    def __init__(self, img_dir, label=0, transform=None):
        self.transform = transform
        self.img_label = int(label)
        self.records = []
        img_dir = Path(img_dir)
        for p in img_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
                self.records.append((str(p), self.img_label, p.name))
        log(f"[DirectoryDataset] dir={img_dir} label={self.img_label} count={len(self.records)}")

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        img_path, label, filename = self.records[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, label, filename


# Model and loss functions
def build_model(model_name: str, num_classes: int, checkpoint_path=None):
    pretrained_flag = False if checkpoint_path else True
    model = timm.create_model(model_name, pretrained=pretrained_flag, num_classes=num_classes)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        new_state = {}
        for k, v in state.items():
            if k.startswith("head") or k.startswith("fc") or "classifier" in k:
                continue
            new_state[k] = v
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        log(f"[CKPT] loaded with missing={len(missing)} unexpected={len(unexpected)}")
    return model

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, weight=None, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = torch.as_tensor(alpha, dtype=torch.float32) if alpha is not None else None
        self.weight = torch.as_tensor(weight, dtype=torch.float32) if weight is not None else None

    def forward(self, logits, target):
        logpt = torch.log_softmax(logits, dim=1)
        pt = torch.exp(logpt)
        onehot = torch.nn.functional.one_hot(target, num_classes=logits.size(1)).float()
        logpt = (logpt * onehot).sum(dim=1)
        pt = (pt * onehot).sum(dim=1)
        loss = -(1 - pt) ** self.gamma * logpt
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)[target]
        elif self.weight is not None:
            loss = loss * self.weight.to(logits.device)[target]
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss

def get_transforms(img_size=256):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tf, test_tf


# Evaluation
@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=2, desc="Evaluating"):
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs, all_fns = [], [], [], []
    for imgs, labels, fns in tqdm(loader, desc=desc):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast():
            logits = model(imgs)
            loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())
        all_fns.extend(list(fns))

    loss = total_loss / len(loader.dataset) if len(loader.dataset) else math.nan
    acc = accuracy_score(all_labels, all_preds) if all_labels else math.nan
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0) if all_labels else math.nan
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0) if all_labels else math.nan
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) if all_labels else math.nan

    auc_per_class = []
    if all_labels and all_probs:
        y_true = np.array(all_labels); y_prob = np.array(all_probs)
        for i in range(num_classes):
            try: auc_i = roc_auc_score((y_true==i).astype(int), y_prob[:, i])
            except ValueError: auc_i = float("nan")
            auc_per_class.append(auc_i)

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes))) if all_labels else np.zeros((num_classes, num_classes), dtype=int)
    return {
        "loss": loss, "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "auc_per_class": auc_per_class, "cm": cm,
        "labels": all_labels, "preds": all_preds, "probs": all_probs, "filenames": all_fns
    }


# Main training and evaluation workflow
def main():
    set_seed(CONFIG["seed"])
    ensure_dirs(CONFIG["out_data_dir"], CONFIG["out_model_dir"], CONFIG["log_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[Env] device={device}")

    train_tf, test_tf = get_transforms(CONFIG["img_size"])

    # Training set: F0 images listed in column 1 plus F1 rows labeled 1 in column 3.
    train_f0_ds = ExcelListDataset(CONFIG["TRAIN_F0_XLSX"], CONFIG["TRAIN_F0_DIR"], label=0, transform=train_tf)
    train_f1_ds = ExcelF1Dataset(CONFIG["TRAIN_F1_XLSX"], CONFIG["TRAIN_F1_DIR"], transform=train_tf)
    train_dataset = ConcatDataset([train_f0_ds, train_f1_ds])

    # External evaluation set: all F0 images in the directory plus F1 rows labeled 1 in column 3.
    test_f0_ds  = DirectoryDataset(CONFIG["TEST_F0_DIR"], label=0, transform=test_tf)
    test_f1_ds  = ExcelF1Dataset(CONFIG["TEST_F1_XLSX"], CONFIG["TEST_F1_DIR"], transform=test_tf)
    test_dataset = ConcatDataset([test_f0_ds, test_f1_ds])

    n0, n1 = len(train_f0_ds), len(train_f1_ds)
    total = n0 + n1
    log(f"[Count] Train F0={n0} F1={n1} total={total} | Test={len(test_dataset)}")

    # Class weights
    if CONFIG["class_weight_mode"] == "auto":
        ce_weights = [ total / (2.0 * max(n0, 1)), total / (2.0 * max(n1, 1)) ]
    elif CONFIG["class_weight_mode"] == "manual":
        ce_weights = CONFIG["manual_class_weights"]
    else:
        ce_weights = None
    if ce_weights is not None:
        log(f"[ClassWeights] CE weights (F0,F1) = {ce_weights}")

    # Optional weighted sampler
    sampler = None
    if CONFIG["use_weighted_sampler"]:
        labels_seq = np.array([0]*n0 + [1]*n1, dtype=np.int64)
        w0 = ce_weights[0] if ce_weights is not None else 1.0
        w1 = ce_weights[1] if ce_weights is not None else 1.0
        per_sample_weights = np.where(labels_seq==0, w0, w1).astype(np.float64)
        sampler = WeightedRandomSampler(
            weights=per_sample_weights,
            num_samples=len(per_sample_weights),
            replacement=True
        )
        log("[Sampler] Using WeightedRandomSampler")

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"],
        shuffle=(sampler is None), sampler=sampler,
        num_workers=CONFIG["num_workers"],
        pin_memory=True, persistent_workers=(CONFIG["num_workers"]>0)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True, persistent_workers=(CONFIG["num_workers"]>0)
    )

    # Model, optimizer, and loss function
    model = build_model(CONFIG["model_name"], CONFIG["num_classes"], CONFIG["checkpoint_path"]).to(device)

    if CONFIG["use_focal_loss"]:
        alpha = CONFIG["focal_alpha"]
        if alpha is None and ce_weights is not None:
            alpha = ce_weights
        criterion = FocalLoss(gamma=CONFIG["focal_gamma"], alpha=alpha, weight=None, reduction="mean")
        log(f"[Loss] FocalLoss(gamma={CONFIG['focal_gamma']}, alpha={alpha})")
    else:
        if ce_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(ce_weights, dtype=torch.float32).to(device))
            log("[Loss] CrossEntropyLoss with class weights")
        else:
            criterion = nn.CrossEntropyLoss()
            log("[Loss] CrossEntropyLoss (unweighted)")

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=CONFIG["step_size"], gamma=CONFIG["gamma"])
    scaler = GradScaler()

    # Training loop
    best_f1 = -1.0
    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        run_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}")
        for imgs, labels, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=float(loss.item()))
        scheduler.step()
        train_epoch_loss = run_loss / len(train_dataset) if len(train_dataset) else math.nan
        log(f"[Train] epoch={epoch} loss={train_epoch_loss:.6f}")

        # Evaluate on the external dataset
        metrics = evaluate(model, test_loader, criterion, device, num_classes=CONFIG["num_classes"], desc=f"Eval e{epoch}")
        log(f"[Test ] epoch={epoch} loss={metrics['loss']:.6f} acc={metrics['acc']:.4f} "
            f"P={metrics['prec']:.4f} R={metrics['rec']:.4f} F1={metrics['f1']:.4f} "
            f"AUC={metrics['auc_per_class']} CM=\n{metrics['cm']}")

        # Save per-image predictions
        df_pred = pd.DataFrame({
            "Filename": metrics["filenames"],
            "True": metrics["labels"],
            "Pred": metrics["preds"],
            "Prob_0": [p[0] for p in metrics["probs"]],
            "Prob_1": [p[1] for p in metrics["probs"]],
        })
        df_pred.to_csv(Path(CONFIG["out_data_dir"]) / f"results_epoch_{epoch}.csv",
                       index=False, encoding="utf-8-sig")

        # Save evaluation metrics
        auc0 = metrics["auc_per_class"][0] if len(metrics["auc_per_class"])>0 else float("nan")
        auc1 = metrics["auc_per_class"][1] if len(metrics["auc_per_class"])>1 else float("nan")
        cm = metrics["cm"]
        met_row = {
            "epoch": epoch,
            "train_loss": train_epoch_loss,
            "test_loss": metrics["loss"],
            "accuracy": metrics["acc"],
            "precision": metrics["prec"],
            "recall": metrics["rec"],
            "f1": metrics["f1"],
            "auc_0": auc0,
            "auc_1": auc1,
            "cm_00": int(cm[0,0]) if cm.shape==(2,2) else None,
            "cm_01": int(cm[0,1]) if cm.shape==(2,2) else None,
            "cm_10": int(cm[1,0]) if cm.shape==(2,2) else None,
            "cm_11": int(cm[1,1]) if cm.shape==(2,2) else None,
            "lr": optimizer.param_groups[0]["lr"],
        }
        pd.DataFrame([met_row]).to_csv(Path(CONFIG["out_data_dir"]) / f"metrics_epoch_{epoch}.csv",
                                       index=False, encoding="utf-8-sig")

        # Save model checkpoints
        # Save checkpoints at the end of each epoch
        # Save the current epoch checkpoint
        epoch_ckpt = Path(CONFIG["out_model_dir"]) / f"epoch_{epoch}.pth"
        torch.save(model.state_dict(), epoch_ckpt)

        # Save the latest checkpoint
        last_ckpt = Path(CONFIG["out_model_dir"]) / "last.pth"
        torch.save(model.state_dict(), last_ckpt)

        # Save the best checkpoint according to the original selection rule
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_ckpt = Path(CONFIG["out_model_dir"]) / "best.pth"
            torch.save(model.state_dict(), best_ckpt)
            log(f"[Model] epoch={epoch} new best.pth saved (macro-F1={best_f1:.4f})")
    log("[Done] Training finished.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"[FATAL] {repr(e)}")
        raise
