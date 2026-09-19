import os
import argparse
from tqdm import tqdm
import torch
import torch.cuda.amp as amp
import matplotlib.pyplot as plt

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


def save_training_plots(history, save_dir):
    """Generates and saves Loss, IoU, and F1 Score graphs."""
    epochs = range(1, len(history['train_loss']) + 1)
    os.makedirs(save_dir, exist_ok=True)

    # 1. Loss Graph
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o', color='crimson')
    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o', color='dodgerblue')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(save_dir, 'loss_graph.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. IoU Graph
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_iou'], label='Train IoU', marker='o', color='crimson')
    plt.plot(epochs, history['val_iou'], label='Val IoU', marker='o', color='dodgerblue')
    plt.title('Training and Validation IoU Score')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(save_dir, 'iou_graph.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. F1 Score Graph
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history['train_f1'], label='Train F1 Score', marker='o', color='crimson')
    plt.plot(epochs, history['val_f1'], label='Val F1 Score', marker='o', color='dodgerblue')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(save_dir, 'f1_score_graph.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Training graphs (Loss, IoU, F1 Score) successfully saved to {save_dir}/")


def save_model_architecture(model, img_size, save_dir, device):
    """Saves a visual representation of the model architecture."""
    try:
        from torchview import draw_graph
        dummy_a = torch.randn(1, 3, img_size, img_size, device=device)
        dummy_b = torch.randn(1, 3, img_size, img_size, device=device)
        model_graph = draw_graph(
            model, 
            input_data=(dummy_a, dummy_b), 
            expand_nested=True, 
            filename=os.path.join(save_dir, "model_architecture"), 
            graph_name="SiameseUNetAttention"
        )
        print(f"🖼️ Model architecture image saved to {save_dir}/model_architecture.png")
    except ImportError:
        print("⚠️ torchview not found. Skipping architecture image generation. (Install via `pip install torchview graphviz`)")
    except Exception as e:
        print(f"⚠️ Could not generate model architecture image: {e}")


def main():
    parser = argparse.ArgumentParser(description="SIH1518 Siamese Change Detection Training")
    parser.add_argument("--data_dir", type=str, default="./data/raw/levir_cd", help="Path to LEVIR-CD root dataset")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--save_dir", type=str, default="./models")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint .pth file to resume training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing Training Pipeline on Device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=4
    )

    # Initialize Model, Loss, Optimizer, Scaler, Scheduler
    model = SiameseUNetAttention(pretrained=True).to(device)
    criterion = HybridChangeDetectionLoss(focal_alpha=0.25, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    # Save Model Architecture Image at start
    save_model_architecture(model, args.img_size, args.save_dir, device)

    start_epoch = 1
    best_val_f1 = 0.0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_f1': [], 'val_f1': []
    }

    # Resume from checkpoint if provided
    if args.resume and os.path.isfile(args.resume):
        print(f"🔄 Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        best_val_f1 = checkpoint.get('best_f1', 0.0)
        if 'history' in checkpoint:
            history = checkpoint['history']
        print(f"✅ Successfully resumed! Starting from Epoch {start_epoch}, Best Val F1: {best_val_f1:.4f}")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n--- Epoch [{epoch}/{args.epochs}] ---")
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device,
            save_vis=True, vis_path=f"./notebooks/vis_epoch_{epoch}.png"
        )

        scheduler.step()

        # Update History tracking
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_metrics['iou'])
        history['val_iou'].append(val_metrics['iou'])
        history['train_f1'].append(train_metrics['f1_score'])
        history['val_f1'].append(val_metrics['f1_score'])

        print(f"Train Loss: {train_loss:.4f} | IoU: {train_metrics['iou']:.4f} | F1: {train_metrics['f1_score']:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | IoU: {val_metrics['iou']:.4f} | F1: {val_metrics['f1_score']:.4f}")

        # Save Checkpoint (Both Latest for resuming and Best Model)
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_f1': best_val_f1,
            'history': history,
        }

        # Save latest checkpoint for easy resume capability
        latest_path = os.path.join(args.save_dir, "latest_checkpoint.pth")
        torch.save(checkpoint_dict, latest_path)

        # Save Best Model Checkpoint
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            checkpoint_dict['best_f1'] = best_val_f1
            best_model_path = os.path.join(args.save_dir, "best_siamese_model.pth")
            torch.save(checkpoint_dict, best_model_path)
            print(f"🏆 Best model checkpoint saved to {best_model_path} (Val F1: {best_val_f1:.4f})")

    # Save final graphs at the end of training
    print("\n🏁 Training completed. Generating final training graphs...")
    save_training_plots(history, args.save_dir)


if __name__ == "__main__":
    main()