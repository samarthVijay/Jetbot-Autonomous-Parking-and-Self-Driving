import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from collections import Counter
from .dataset import JetbotDataset, FOLDER_TO_CLASS
from .model import ParkingNet, CLASS_NAMES


class FocalLoss(nn.Module):
    """
    Focal Loss for classification.
    Down-weights easy examples so training focuses on hard/confused pairs.
    Reference: https://arxiv.org/abs/1708.02002
    gamma=2 is the standard setting. alpha handles class imbalance.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha  # per-class weight tensor
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        return focal.mean() if self.reduction == "mean" else focal.sum()


def compute_class_weights(dataset, num_classes):
    """
    Inverse-frequency class weights so the overrepresented 'blocked' class
    (130 obstacle + 100 occupied = 230) doesn't dominate training.
    """
    counts = Counter(label for _, label in dataset.samples)
    total = sum(counts.values())
    weights = []
    for i in range(num_classes):
        n = counts.get(i, 1)
        weights.append(total / (num_classes * n))
    return torch.tensor(weights, dtype=torch.float32)


def train_model(
    data_dir="dataset",
    epochs=35,
    batch_size=16,
    lr=5e-4,
    backbone="mobilenet_v2",
    save_path="best_model.pth",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    full_dataset = JetbotDataset(data_dir=data_dir, is_train=True)
    if len(full_dataset) < 20:
        print("Error: Dataset too small — collect at least 20 images per class first.")
        return

    val_size = max(int(len(full_dataset) * 0.2), 1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_dataset.dataset.is_train = False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {train_size} | Val: {val_size}")

    model = ParkingNet(num_classes=len(CLASS_NAMES), backbone=backbone).to(device)

    class_weights = compute_class_weights(full_dataset, len(CLASS_NAMES)).to(device)
    print(f"Class weights: { {CLASS_NAMES[i]: round(w.item(), 3) for i, w in enumerate(class_weights)} }")
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    patience_counter = 0
    PATIENCE = 10

    print(f"\nTraining {backbone} for up to {epochs} epochs...\n")
    print(f"{'Epoch':>6} {'LR':>9} {'TrainLoss':>10} {'TrainAcc':>9} {'ValAcc':>8} {'Best':>6}")
    print("-" * 56)

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * images.size(0)
            correct  += (outputs.argmax(1) == labels).sum().item()
            total    += labels.size(0)

        train_loss = run_loss / total
        train_acc  = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                val_correct += (model(images).argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        val_acc  = val_correct / val_total if val_total > 0 else 0.0
        cur_lr   = scheduler.get_last_lr()[0]
        is_best  = val_acc >= best_val_acc
        marker   = " <--" if is_best else ""

        print(f"{epoch:>6} {cur_lr:>9.6f} {train_loss:>10.4f} {train_acc:>9.4f} {val_acc:>8.4f}{marker}")

        if is_best:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs).")
                break

        scheduler.step()

    print(f"\nBest val acc: {best_val_acc:.4f} | Saved to '{save_path}'")

    print("\nPer-class accuracy on validation set:")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    class_correct = Counter()
    class_total   = Counter()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(1)
            for p, t in zip(preds.cpu(), labels.cpu()):
                class_total[t.item()]   += 1
                class_correct[t.item()] += int(p == t)
    for i, name in enumerate(CLASS_NAMES):
        n = class_total.get(i, 0)
        c = class_correct.get(i, 0)
        acc_str = f"{c/n:.3f}" if n > 0 else "  n/a"
        print(f"  [{i}] {name:<12}: {c:>3}/{n:<3} = {acc_str}")

    print("\nExporting ONNX...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.export_onnx("best_model.onnx")
    print("Done.")


if __name__ == "__main__":
    train_model()
