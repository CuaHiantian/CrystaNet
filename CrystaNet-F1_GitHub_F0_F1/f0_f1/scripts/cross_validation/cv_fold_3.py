import os
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

# Define the guided image dataset
class GuidedImageDataset(Dataset):
    def __init__(self, guide_file, img_dir, label, transform=None):
        self.transform = transform
        self.data = []

        df = pd.read_excel(guide_file)

        # For the liver fibrosis guide, retain rows labeled as stage F1 (value 1).
        if 'liver_fibrosis_list' in str(guide_file).lower():
            df = df[df.iloc[:, 2] == 1]

        # Assign all images with the same participant ID to the same split (second column).
        id_to_split = {}
        for sample_id in df.iloc[:, 1].unique():
            sample_id = int(sample_id)
            id_to_split[sample_id] = 'test' if sample_id % 5 == 2 else 'train'

        # Keep records whose image files exist.
        not_found = 0
        for _, row in df.iterrows():
            filename = row[0]
            sample_id = int(row[1])
            split = id_to_split[sample_id]
            img_path = os.path.join(img_dir, filename)
            if os.path.exists(img_path):
                self.data.append((img_path, label, filename, split))
            else:
                not_found += 1

        print(f"[{os.path.basename(guide_file)}] images not found: {not_found}")

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
    log_dir = 'epoch-logs'
    os.makedirs(log_dir, exist_ok=True)
    def log(msg):
        print(msg)
        with open(os.path.join(log_dir, 'training_log.txt'), 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now()} - {msg}\n")

    model_name = 'convnext_base'
    num_classes = 2

    checkpoint_path = str(PRETRAINED_WEIGHTS)  # Configurable pretrained checkpoint path.
    # Create the backbone; the classification head will be replaced below.
    base_model = timm.create_model(model_name, pretrained=False)

    # Load pretrained parameters.
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    # Remove pretrained classification-head parameters that do not match this task.
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.fc')}

    # Load the remaining pretrained parameters, excluding head.fc.
    base_model.load_state_dict(state_dict, strict=False)

    # Replace the classification head for binary classification.
    in_features = base_model.head.fc.in_features
    base_model.head.fc = nn.Linear(in_features, num_classes)

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize input images.
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Apply the same resize to evaluation images.
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    f1_dir = DATA_ROOT / "internal" / "f1_images"
    f0_dir = DATA_ROOT / "internal" / "f0_images"
    f1_guide = DATA_ROOT / "guides" / "liver_fibrosis_list.xlsx"
    f0_guide = DATA_ROOT / "guides" / "f0_image_list.xlsx"

    f1_dataset = GuidedImageDataset(f1_guide, f1_dir, label=1, transform=train_transform)
    f0_dataset = GuidedImageDataset(f0_guide, f0_dir, label=0, transform=train_transform)

    f1_train = f1_dataset.get_split('train')
    f0_train = f0_dataset.get_split('train')
    f1_test = f1_dataset.get_split('test')
    f0_test = f0_dataset.get_split('test')

    train_dataset = ConcatDataset([f1_train, f0_train])
    test_dataset = ConcatDataset([f1_test, f0_test])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = base_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-7, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    scaler = GradScaler()

    os.makedirs('epoch-data-2', exist_ok=True)
    os.makedirs('epoch-models-2', exist_ok=True)

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
        auc = []
        for i in range(num_classes):
            try:
                score = roc_auc_score((np.array(all_labels) == i).astype(int), np.array(all_probs)[:, i])
            except ValueError:
                score = float('nan')  # Keep NaN when an AUC value cannot be calculated.
            auc.append(score)

        return test_loss, accuracy, precision, recall, f1, cm, auc, all_labels, all_preds, all_probs, all_filenames

    num_epochs = 5
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
        print(f'Epoch {epoch + 1}/{num_epochs}, Training loss: {epoch_loss:.4f}')

        test_loss, accuracy, precision, recall, f1, cm, auc, all_labels, all_preds, all_probs, all_filenames = evaluate_model(
            model, test_loader, criterion, device
        )

        results_df = pd.DataFrame({
            'Filename': all_filenames,
            'True Label': all_labels,
            'Predicted Label': all_preds,
            'Probabilities': list(all_probs)
        })
        results_df.to_csv(f'epoch-data-2/results_epoch_{epoch + 1}.csv', index=False, encoding='utf-8-sig')

        print(f'Epoch {epoch + 1}/{num_epochs}, Evaluation loss: {test_loss:.4f}')
        print(f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        print('Confusion matrix:')
        print(cm)
        print('AUC by class:', auc)

        metrics = {
            'Test Loss': test_loss,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Confusion matrix': cm,
            'AUC': auc
        }
        pd.DataFrame([metrics]).to_csv(f'epoch-data-2/metrics_epoch_{epoch + 1}.csv', index=False, encoding='utf-8-sig')
        torch.save(model.state_dict(), f'epoch-models-2/model_epoch_{epoch + 1}.pth')

