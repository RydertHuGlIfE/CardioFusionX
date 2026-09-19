
import csv
import shutil
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]   ##root to pythonpath


from eng_dataset import ECGDataset
from model import ECGCNN 


#configs  

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001


#nogpu-cpu-killer-69000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


#datasetloader

train_dataset = ECGDataset("train_split.csv")
val_dataset = ECGDataset("val_split.csv")

positive_counts = torch.tensor(
    train_dataset.df[train_dataset.label_columns].sum(axis=0).to_numpy(),
    dtype=torch.float32,
)
negative_counts = len(train_dataset) - positive_counts
pos_weight = torch.where(
    positive_counts > 0,
    negative_counts / positive_counts,
    torch.ones_like(positive_counts),
).clamp(max=20)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


#model,loss.OPTIMIZE

model = ECGCNN(num_classes=94).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight.to(DEVICE)
)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

latest_dir = Path("experiments/latest")
best_dir = Path("experiments/best")
history_path = Path("experiments/training_history.csv")
latest_dir.mkdir(parents=True, exist_ok=True)
best_dir.mkdir(parents=True, exist_ok=True)

for path in latest_dir.iterdir():
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()

best_checkpoint_path = best_dir / "model.pth"
best_val_loss = float("inf")
if best_checkpoint_path.exists():
    best_checkpoint = torch.load(
        best_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(best_checkpoint, dict):
        best_val_loss = best_checkpoint.get("val_loss", float("inf"))

with history_path.open("w", newline="") as history_file:
    history_writer = csv.writer(history_file)
    history_writer.writerow(["epoch", "train_loss", "val_loss"])



#train and validate
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    train_progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} - Train",
    )
    for signals, labels in train_progress:
        signals = signals.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(signals)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * signals.size(0)
        train_progress.set_postfix(batch_loss=f"{loss.item():.4f}")

    train_loss /= len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        val_progress = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS} - Validation",
        )
        for signals, labels in val_progress:
            signals = signals.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(signals)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * signals.size(0)
            val_progress.set_postfix(batch_loss=f"{loss.item():.4f}")

    val_loss /= len(val_loader.dataset)
    
    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, latest_dir / "model.pth")

    if val_loss < best_val_loss:
        torch.save(checkpoint, best_checkpoint_path)
        best_val_loss = val_loss

    with history_path.open("a", newline="") as history_file:
        history_writer = csv.writer(history_file)
        history_writer.writerow([epoch + 1, train_loss, val_loss])
