import os
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import timm
from tqdm import tqdm
from torch.cuda.amp import autocast


# Portable paths; override with CRYSTANET_DATA_ROOT, CRYSTANET_WEIGHTS_ROOT,
# or CRYSTANET_OUTPUT_ROOT. The defaults are relative to the repository.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("CRYSTANET_DATA_ROOT", str(PROJECT_ROOT / "data"))).expanduser()
WEIGHTS_ROOT = Path(os.environ.get("CRYSTANET_WEIGHTS_ROOT", str(PROJECT_ROOT / "weights"))).expanduser()
OUTPUT_ROOT = Path(os.environ.get("CRYSTANET_OUTPUT_ROOT", str(PROJECT_ROOT / "outputs"))).expanduser()

# -----------------------------
# Dataset definition
# -----------------------------
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

# -----------------------------
# Load the classification model
# -----------------------------
def load_model(model_path, num_classes=2, device='cuda'):
    model_name = 'convnext_base'
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model

# -----------------------------
# Inference function
# -----------------------------
def infer(model, loader, device='cuda'):
    model.eval()
    all_labels, all_preds, all_probs, all_filenames = [], [], [], []
    with torch.no_grad():
        for imgs, labels, filenames in tqdm(loader, desc="Running inference"):
            imgs = imgs.to(device)
            with autocast():
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_filenames.extend(filenames)
    return all_labels, all_preds, all_probs, all_filenames

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = WEIGHTS_ROOT / "hc_vs_lf_final.pth"

    # Image directories and Excel guide paths
    healthy_excel = DATA_ROOT / "external_shijiazhuang" / "healthy" / "guide.xlsx"
    healthy_imgdir = DATA_ROOT / "external_shijiazhuang" / "healthy" / "images"

    fibrosis_excel = DATA_ROOT / "external_shijiazhuang" / "fibrosis" / "guide.xlsx"
    fibrosis_imgdir = DATA_ROOT / "external_shijiazhuang" / "fibrosis" / "images"

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    healthy_ds = GuidedImageDataset(healthy_excel, healthy_imgdir, label=0, transform=transform)
    fibrosis_ds = GuidedImageDataset(fibrosis_excel, fibrosis_imgdir, label=1, transform=transform)
    test_dataset = torch.utils.data.ConcatDataset([healthy_ds, fibrosis_ds])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Healthy images: {len(healthy_ds)}, Fibrosis images: {len(fibrosis_ds)}, Total images: {len(test_dataset)}")

    # Load the classification model
    model = load_model(model_path, num_classes=2, device=device)

    # Run inference on the external dataset
    labels, preds, probs, filenames = infer(model, test_loader, device)

    # Save prediction results
    output_dir = OUTPUT_ROOT / "external_shijiazhuang"
    os.makedirs(output_dir, exist_ok=True)

    result_df = pd.DataFrame({
        "Filename": filenames,
        "True Label": labels,
        "Predicted Label": preds,
        "Prob_Healthy": [p[0] for p in probs],
        "Prob_Fibrosis": [p[1] for p in probs]
    })
    result_path = os.path.join(output_dir, "shijiazhuang_predictions.csv")
    result_df.to_csv(result_path, index=False, encoding='utf-8-sig')
    print(f"Inference completed. Results saved to: {result_path}")
