import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from models.multimodal import MultiModalModel

class MultiModalDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        spectrum = torch.tensor(row[1:-1].values, dtype=torch.float32)
        label = torch.tensor(row[-1], dtype=torch.long)
        return image, spectrum, label

# Load dataset
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

dataset = MultiModalDataset('data/sample_spectra.csv', 'data/sample_images', transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiModalModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    total_loss = 0
    for img, spec, lbl in loader:
        img, spec, lbl = img.to(device), spec.to(device), lbl.to(device)
        optimizer.zero_grad()
        outputs = model(img, spec)
        loss = criterion(outputs, lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "model.pth")
