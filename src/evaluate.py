import os
import argparse
import torch
from tqdm import tqdm

from dataset import LEVIRCDDataset
from torch.utils.data import DataLoader
from model import SiameseUNetAttention
from loss import HybridChangeDetectionLoss
from utils import MetricTracker, save_visualization_sample


@torch.no_grad()
def run_evaluation(data_dir: str, weights_path: str, batch_size: int = 16, img_size: int = 256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running Comprehensive Test Split Evaluation on: {device}")

    # 1. Load Test Dataset
    test_dir = os.path.join(data_dir, "test")
    test_dataset = LEVIRCDDataset(test_dir, img_size=img_size, is_train=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # 2. Initialize Model and Load Weights
    model = SiameseUNetAttention(pretrained=False).to(device)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = HybridChangeDetectionLoss()
    tracker = MetricTracker(threshold=0.5)
    running_loss = 0.0

    os.makedirs("./notebooks/test_visualizations", exist_ok=True)

    print(f"📊 Evaluating {len(test_dataset)} bitemporal test pairs...")
    for idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
        img_a = batch["img_a"].to(device, non_blocking=True)
        img_b = batch["img_b"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda"):
            logits = model(img_a, img_b)
            loss = criterion(logits, masks)

        running_loss += loss.item() * img_a.size(0)
        tracker.update(logits, masks)

        # Save sample visual predictions for demo presentation
        if idx < 5:
            save_path = f"./notebooks/test_visualizations/sample_{idx + 1}.png"
            save_visualization_sample(img_a[0], img_b[0], masks[0], logits[0], save_path)

    # 3. Compute Metrics
    avg_loss = running_loss / len(test_dataset)
    results = tracker.compute()

    print("\n" + "=" * 50)
    print("🎯 FINAL TEST SET BENCHMARK PERFORMANCE")
    print("=" * 50)
    print(f"  • Test Loss:    {avg_loss:.4f}")
    print(f"  • Precision:    {results['precision'] * 100:.2f}%")
    print(f"  • Recall:       {results['recall'] * 100:.2f}%")
    print(f"  • F1-Score:     {results['f1_score'] * 100:.2f}%")
    print(f"  • IoU (Jaccard): {results['iou'] * 100:.2f}%")
    print("=" * 50)
    print("🖼️ Visual qualitative comparisons saved to: ./notebooks/test_visualizations/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Change Detection Model on Test Set")
    parser.add_argument("--data_dir", type=str, default="./data/raw/levir_cd")
    parser.add_argument("--weights", type=str, default="./models/best_siamese_model.pth")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()

    run_evaluation(args.data_dir, args.weights, args.batch_size, args.img_size)