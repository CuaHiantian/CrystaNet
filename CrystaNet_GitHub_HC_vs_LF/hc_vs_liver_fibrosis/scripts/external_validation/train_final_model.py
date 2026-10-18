import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
import timm
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
from tqdm import tqdm

# Configurable manual class weights
# - Set to None to use unweighted cross-entropy loss.
# - For example, [1.0, 3.0] assigns weights 1.0 and 3.0 to classes 0 and 1.
manual_class_weights = [1.0, 1.3]


# Portable paths; override with CRYSTANET_DATA_ROOT, CRYSTANET_WEIGHTS_ROOT,
# or CRYSTANET_OUTPUT_ROOT. The defaults are relative to the repository.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("CRYSTANET_DATA_ROOT", str(PROJECT_ROOT / "data"))).expanduser()
WEIGHTS_ROOT = Path(os.environ.get("CRYSTANET_WEIGHTS_ROOT", str(PROJECT_ROOT / "weights"))).expanduser()
OUTPUT_ROOT = Path(os.environ.get("CRYSTANET_OUTPUT_ROOT", str(PROJECT_ROOT / "outputs"))).expanduser()

# Dataset implementation
class GuidedImageDataset(Dataset):
    def __init__(self, guide_file, img_dir, label, transform=None):
        self.transform = transform
        self.data = []
        df = pd.read_excel(guide_file)
        for _, row in df.iterrows():
            filename = str(row[0])
            img_path = os.path.join(img_dir, filename)
            if os.path.exists(img_path):
                self.data.append((img_path, label, filename))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label, filename = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, filename

if __name__ == '__main__':
    log_dir = OUTPUT_ROOT / 'epoch-logs'
    os.makedirs(log_dir, exist_ok=True)

    def log(msg):
        print(msg)
        with open(os.path.join(log_dir, 'training_log.txt'), 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now()} - {msg}\n")

    model_name = 'convnext_base'
    num_classes = 2
    checkpoint_path = WEIGHTS_ROOT / "convnext_pretrained" / "pytorch_model.bin"

    # Initialize model and load pretrained weights
    base_model = timm.create_model(model_name, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.fc')}
    base_model.load_state_dict(state_dict, strict=False)
    in_features = base_model.head.fc.in_features
    base_model.head.fc = nn.Linear(in_features, num_classes)

    # Image transformations
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load training dataset
    healthy_train_dataset = GuidedImageDataset(
        guide_file=DATA_ROOT / "internal" / "healthy" / "guide.xlsx",
        img_dir=DATA_ROOT / "internal" / "healthy" / "images",
        label=0,
        transform=train_transform)

    fibrosis_train_dataset = GuidedImageDataset(
        guide_file=DATA_ROOT / "internal" / "fibrosis" / "guide.xlsx",
        img_dir=DATA_ROOT / "internal" / "fibrosis" / "images",
        label=1,
        transform=train_transform)

    train_dataset = ConcatDataset([healthy_train_dataset, fibrosis_train_dataset])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

    # Load validation dataset
    healthy_val_dataset = GuidedImageDataset(
        guide_file=DATA_ROOT / "external_zengcheng" / "healthy" / "guide.xlsx",
        img_dir=DATA_ROOT / "external_zengcheng" / "healthy" / "images",
        label=0,
        transform=test_transform)

    fibrosis_val_dataset = GuidedImageDataset(
        guide_file=DATA_ROOT / "external_zengcheng" / "fibrosis" / "guide.xlsx",
        img_dir=DATA_ROOT / "external_zengcheng" / "fibrosis" / "images",
        label=1,
        transform=test_transform)

    test_dataset = ConcatDataset([healthy_val_dataset, fibrosis_val_dataset])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # Select device and move the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = base_model.to(device)

    # Optional class weights
    def build_ce_criterion(weights, n_classes, device_):
        """Build weighted cross-entropy; use unweighted loss for invalid weights."""
        if weights is None:
            log("[Loss] Using uniform class weights")
            return nn.CrossEntropyLoss()

        # Validate the user-provided class weights
        try:
            if len(weights) != n_classes:
                raise ValueError(f"manual_class_weights must contain {n_classes} weights; received {len(weights)}: {weights}")
            if any((w is None) or (float(w) <= 0) for w in weights):
                raise ValueError(f"manual_class_weights must contain positive values: {weights}")
            w_tensor = torch.tensor([float(w) for w in weights], dtype=torch.float32, device=device_)
            log(f"[Loss] Using manual class weights: {list(map(float, weights))}")
            return nn.CrossEntropyLoss(weight=w_tensor)
        except Exception as e:
            log(f"[Loss][WARN] Invalid class weights; using unweighted cross-entropy. Reason: {e}")
            return nn.CrossEntropyLoss()

    criterion = build_ce_criterion(manual_class_weights, num_classes, device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-7, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    scaler = GradScaler()

    os.makedirs(OUTPUT_ROOT / 'final_training' / 'epoch_data', exist_ok=True)
    os.makedirs(OUTPUT_ROOT / 'final_training' / 'epoch_models', exist_ok=True)

    def evaluate_model(model, test_loader, criterion, device='cuda'):
        model.eval()
        running_loss = 0.0
        all_labels, all_preds, all_probs, all_filenames = [], [], [], []
        with torch.no_grad():
            for inputs, labels, filenames in tqdm(test_loader, desc='Evaluating'):
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs)
                all_filenames.extend(filenames)

        test_loss = running_loss / len(test_loader.dataset)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        cm = confusion_matrix(all_labels, all_preds)
        auc = [roc_auc_score((np.array(all_labels) == i).astype(int), np.array(all_probs)[:, i]) for i in range(num_classes)]

        return test_loss, accuracy, precision, recall, f1, cm, auc, all_labels, all_preds, all_probs, all_filenames

    num_epochs = 8
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels, filenames in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
        scheduler.step()

        epoch_loss = running_loss / len(train_dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')

        test_loss, accuracy, precision, recall, f1, cm, auc, all_labels, all_preds, all_probs, all_filenames = evaluate_model(
            model, test_loader, criterion, device
        )

        results_df = pd.DataFrame({
            'Filename': all_filenames,
            'True Label': all_labels,
            'Predicted Label': all_preds,
            'Probabilities': list(all_probs)
        })
        results_df.to_csv(OUTPUT_ROOT / 'final_training' / 'epoch_data' / f'results_epoch_{epoch + 1}-final.csv', index=False, encoding='utf-8-sig')

        print(f'Epoch {epoch + 1}/{num_epochs}, Test Loss: {test_loss:.4f}')
        print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        print('Confusion Matrix:')
        print(cm)
        print('AUC for each class:', auc)

        metrics = {
            'Test Loss': test_loss,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Confusion Matrix': cm,
            'AUC': auc
        }
        pd.DataFrame([metrics]).to_csv(OUTPUT_ROOT / 'final_training' / 'epoch_data' / f'metrics_epoch_{epoch + 1}-final.csv', index=False, encoding='utf-8-sig')
        torch.save(model.state_dict(), OUTPUT_ROOT / 'final_training' / 'epoch_models' / f'model_epoch_{epoch + 1}-final.pth')
