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


# Portable paths; override with CRYSTANET_DATA_ROOT, CRYSTANET_WEIGHTS_ROOT,
# or CRYSTANET_OUTPUT_ROOT. The defaults are relative to the repository.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("CRYSTANET_DATA_ROOT", str(PROJECT_ROOT / "data"))).expanduser()
WEIGHTS_ROOT = Path(os.environ.get("CRYSTANET_WEIGHTS_ROOT", str(PROJECT_ROOT / "weights"))).expanduser()
OUTPUT_ROOT = Path(os.environ.get("CRYSTANET_OUTPUT_ROOT", str(PROJECT_ROOT / "outputs"))).expanduser()

# Dataset implementation
class GuidedImageDataset(Dataset):
    def __init__(self, guide_file, img_dir, label, transform=None, test_split=True):
        self.transform = transform
        self.data = []
        df = pd.read_excel(guide_file)
        for _, row in df.iterrows():
            filename, number = row[0], row[1]
            img_path = os.path.join(img_dir, filename)
            if os.path.exists(img_path):
                split = 'test' if number % 5 == 3 else 'train'
                self.data.append((img_path, label, filename, split))

    def get_split(self, split):
        split_data = [item for item in self.data if item[3] == split]
        return GuidedImageDatasetSubset(split_data, self.transform)

class GuidedImageDatasetSubset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label, filename, _ = self.data[idx]
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

    checkpoint_path = WEIGHTS_ROOT / "convnext_pretrained" / "pytorch_model.bin"  # Path to the pretrained ConvNeXt checkpoint
    # Initialize the pretrained backbone before replacing the classifier
    base_model = timm.create_model(model_name, pretrained=False)

    # Load pretrained model weights
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    # Exclude classifier parameters with incompatible output dimensions
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.fc')}

    # Load backbone parameters, excluding head.fc
    base_model.load_state_dict(state_dict, strict=False)

    # Replace the classification head
    in_features = base_model.head.fc.in_features
    base_model.head.fc = nn.Linear(in_features, num_classes)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Use the configured image preprocessing size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Apply the evaluation image preprocessing
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    healthy_dir = DATA_ROOT / "internal" / "healthy" / "images"
    fibrosis_dir = DATA_ROOT / "internal" / "fibrosis" / "images"
    healthy_guide = DATA_ROOT / "internal" / "healthy" / "guide.xlsx"
    fibrosis_guide = DATA_ROOT / "internal" / "fibrosis" / "guide.xlsx"

    healthy_dataset = GuidedImageDataset(healthy_guide, healthy_dir, label=0, transform=train_transform)
    fibrosis_dataset = GuidedImageDataset(fibrosis_guide, fibrosis_dir, label=1, transform=train_transform)

    healthy_train = healthy_dataset.get_split('train')
    fibrosis_train = fibrosis_dataset.get_split('train')
    healthy_test = healthy_dataset.get_split('test')
    fibrosis_test = fibrosis_dataset.get_split('test')

    train_dataset = ConcatDataset([healthy_train, fibrosis_train])
    test_dataset = ConcatDataset([healthy_test, fibrosis_test])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = base_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    scaler = GradScaler()

    os.makedirs(OUTPUT_ROOT / 'epoch-data', exist_ok=True)
    os.makedirs(OUTPUT_ROOT / 'epoch-models', exist_ok=True)

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

    num_epochs = 6
    best_accuracy = 0.0

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
        results_df.to_csv(f'epoch-data/results_epoch_{epoch + 1}-25w-256-4.csv', index=False, encoding='utf-8-sig')

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
        pd.DataFrame([metrics]).to_csv(f'epoch-data/metrics_epoch_{epoch + 1}-25w-256-4.csv', index=False, encoding='utf-8-sig')
        torch.save(model.state_dict(), f'epoch-models/model_epoch_{epoch + 1}-25w-256-4.pth')

