import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from pathlib import Path
import json
import time

from src.training.dataset import PathMNISTDataset
from src.training.transforms import train_transform, val_transform

# ── Config ────────────────────────────────────────────────
NPZ_PATH    = "data/raw/pathmnist.npz"
MODELS_DIR  = Path("models")
NUM_CLASSES = 9
BATCH_SIZE  = 64
EPOCHS      = 15
LR          = 1e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR.mkdir(exist_ok=True)

# ── Data ──────────────────────────────────────────────────
train_ds = PathMNISTDataset(NPZ_PATH, "train", transform=train_transform)
val_ds   = PathMNISTDataset(NPZ_PATH, "val",   transform=val_transform)

train_loader = DataLoader(train_ds, 
                          batch_size=BATCH_SIZE, 
                          shuffle=True,  
                          num_workers=2)

val_loader   = DataLoader(val_ds,   
                          batch_size=BATCH_SIZE, 
                          shuffle=False, 
                          num_workers=2)


# ── Model ─────────────────────────────────────────────────
model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()


# ── Training loop ─────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs)
            loss  = criterion(preds, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(imgs)
            correct    += (preds.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


best_val_acc = 0.0
history = []

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss,   val_acc   = run_epoch(val_loader,   train=False)
    scheduler.step()

    elapsed = time.time() - t0
    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
          f"val loss {val_loss:.4f} acc {val_acc:.4f} | {elapsed:.1f}s")

    history.append({"epoch": epoch, 
                    "train_loss": train_loss,
                    "train_acc": train_acc, 
                    "val_loss": val_loss, 
                    "val_acc": val_acc})

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODELS_DIR / "best_model.pth")
        print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")

# Save final model + metadata
torch.save(model.state_dict(), MODELS_DIR / "last_model.pth")
with open(MODELS_DIR / "training_history.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"\nDone. Best val acc: {best_val_acc:.4f}")