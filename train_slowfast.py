import os
import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorchvideo.models.hub import slowfast_r50
from PIL import Image
from tqdm import tqdm

# ----------------------------------
# CONFIG
# ----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
VAL_CSV = os.path.join(BASE_DIR, "val.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 2
BATCH_SIZE = 2
EPOCHS = 40
FRAMES_PER_CLIP = 32
ALPHA = 4  # SlowFast temporal ratio

print("Using device:", DEVICE)

# ----------------------------------
# DATASET
# ----------------------------------
class VideoDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, header=None)

        self.transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.45, 0.45, 0.45],
        std=[0.225, 0.225, 0.225]
    )
])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip_path, label = self.data.iloc[idx]
        clip_path = os.path.join(BASE_DIR, clip_path)

        
        # frames = sorted(os.listdir(clip_path))
        # indices = torch.linspace(0, len(frames)-1, FRAMES_PER_CLIP).long()
        # frames = [frames[i] for i in indices]
        frames = sorted(os.listdir(clip_path))

        if len(frames) < FRAMES_PER_CLIP:
            frames = frames + [frames[-1]] * (FRAMES_PER_CLIP - len(frames))

        indices = torch.linspace(
            0, len(frames) - 1,
            steps=FRAMES_PER_CLIP
            ).long()

        frames = [frames[i] for i in indices]

        video = []
        for frame in frames:
            img = Image.open(os.path.join(clip_path, frame)).convert("RGB")
            img = self.transform(img)
            video.append(img)

        video = torch.stack(video)  # (T, C, H, W)
        video = video.permute(1, 0, 2, 3)  # (C, T, H, W)

        # Create SlowFast pathways
        # Ensure exactly 64 frames
        fast_pathway = video[:, :FRAMES_PER_CLIP, :, :]

# Slow pathway derived from fast
        slow_pathway = fast_pathway[:, ::ALPHA, :, :]

        return [slow_pathway, fast_pathway], torch.tensor(label)


# ----------------------------------
# LOAD DATA
# ----------------------------------
train_dataset = VideoDataset(TRAIN_CSV)
val_dataset = VideoDataset(VAL_CSV)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ----------------------------------
# LOAD PRETRAINED SLOWFAST
# ----------------------------------
model = slowfast_r50(pretrained=True)
# Freeze everything first
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.blocks[-1].proj = nn.Linear(
    model.blocks[-1].proj.in_features,
    NUM_CLASSES
)

# Unfreeze classifier
for param in model.blocks[-1].proj.parameters():
    param.requires_grad = True

# 🔥 ALSO unfreeze last 2 blocks
for name, param in model.named_parameters():
    if "blocks.4" in name or "blocks.3" in name:
        param.requires_grad = True

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

# ----------------------------------
# TRAIN LOOP
# ----------------------------------
start_epoch = 0
best_acc = 0

if os.path.exists("slowfast_checkpoint.pth"):
    print("🔄 Resuming from checkpoint...")
    checkpoint = torch.load("slowfast_checkpoint.pth", map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_acc = checkpoint["best_acc"]

    print(f"Resumed from epoch {start_epoch}")

for epoch in range(start_epoch, EPOCHS):

    model.train()
    total_loss = 0

    for videos, labels in tqdm(train_loader):
        slow, fast = videos
        slow = slow.to(DEVICE)
        fast = fast.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model([slow, fast])
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Training Loss: {total_loss:.4f}")

    # ---------------- Validation ----------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for videos, labels in val_loader:
            slow, fast = videos
            slow = slow.to(DEVICE)
            fast = fast.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model([slow, fast])
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")

    # Save best model
    if acc > best_acc:
        best_acc = acc

        checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc
    }

        torch.save(checkpoint, "slowfast_checkpoint.pth")
        print("🔥 Checkpoint saved!")

print("\nTraining Complete!")
print("Best Validation Accuracy:", best_acc)