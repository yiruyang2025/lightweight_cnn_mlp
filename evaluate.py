import torch
from models.multimodal import MultiModalModel
from train import MultiModalDataset, transform
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

dataset = MultiModalDataset('data/sample_spectra.csv', 'data/sample_images', transform)
loader = DataLoader(dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalModel().to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for img, spec, lbl in loader:
        img, spec = img.to(device), spec.to(device)
        outputs = model(img, spec)
        _, preds = torch.max(outputs, 1)
        y_true.extend(lbl.numpy())
        y_pred.extend(preds.cpu().numpy())

print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"Weighted F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")
