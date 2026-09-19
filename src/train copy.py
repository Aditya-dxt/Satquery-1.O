import os
import argparse
from tqdm import tqdm
import torch
import torch.cuda.amp as amp

from dataset import get_dataloaders
from model import SiameseUNetAttention
from loss import HybridChangeDetectionLoss
from utils import calculate_metrics, save_visualization_sample, MetricTracker


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    total_metrics = {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'iou': 0.0}

    for batch in tqdm(dataloader, desc="Training", leave=False):
        img_a = batch['img_a'].to(device, non_blocking=True)
        img_b = batch['img_b'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Automatic Mixed Precision (FP16)
        with torch.amp.autocast('cuda'):
            logits = model(img_a, img_b)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * img_a.size(0)
        metrics = calculate_metrics(logits.detach(), masks)
        for k in total_metrics:
            total_metrics[k] += metrics[k] * img_a.size(0)

    num_samples = len(dataloader.dataset)
    epoch_loss = running_loss / num_samples
    avg_metrics = {k: v / num_samples for k, v in total_metrics.items()}
    return epoch_loss, avg_metrics

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, save_vis: bool = False, vis_path: str = ""):
    model.eval()
    running_loss = 0.0
    tracker = MetricTracker()

    for idx, batch in enumerate(tqdm(dataloader, desc="Validation", leave=False)):
        img_a = batch['img_a'].to(device, non_blocking=True)
        img_b = batch['img_b'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            logits = model(img_a, img_b)
            loss = criterion(logits, masks)

        running_loss += loss.item() * img_a.size(0)
        tracker.update(logits, masks)

        if save_vis and idx == 0:
            save_visualization_sample(img_a[0], img_b[0], masks[0], logits[0], vis_path)

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = tracker.compute()
    return epoch_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="SIH1518 Siamese Change Detection Training")
    parser.add_argument("--data_dir", type=str, default="./data/raw/levir_cd", help="Path to LEVIR-CD root dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="./models")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Training Pipeline on Device: {device}")

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=4
    )

    # Initialize Model, Loss, Optimizer, Scaler
    model = SiameseUNetAttention(pretrained=True).to(device)
    criterion = HybridChangeDetectionLoss(focal_alpha=0.25, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    # scaler = amp.GradScaler()
    scaler = torch.amp.GradScaler('cuda')

    best_val_f1 = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch [{epoch}/{args.epochs}] ---")
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device,
            save_vis=True, vis_path=f"./notebooks/vis_epoch_{epoch}.png"
        )

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | IoU: {train_metrics['iou']:.4f} | F1: {train_metrics['f1_score']:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | IoU: {val_metrics['iou']:.4f} | F1: {val_metrics['f1_score']:.4f}")

        # Save Best Model Checkpoint
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            best_model_path = os.path.join(args.save_dir, "best_siamese_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_val_f1,
            }, best_model_path)
            print(f"🏆 Best model checkpoint saved to {best_model_path} (Val F1: {best_val_f1:.4f})")


if __name__ == "__main__":
    main()