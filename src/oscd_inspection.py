import os
from pathlib import Path
from tqdm import tqdm

def inspect_oscd_split_dataset(data_dir):
    """
    Inspects the OSCD dataset structured with separate 'images' and 'train_labels' directories.
    """
    root_path = Path(data_dir)
    print("=" * 60)
    print(f"🔎 STARTING OSCD DATASET INSPECTION FOR: {root_path.resolve()}")
    print("=" * 60)

    # 1. Define paths for the split structure
    images_dir = root_path / "images"
    labels_dir = root_path / "train_labels"

    if not images_dir.exists():
        print(f"❌ Error: 'images' directory not found at {images_dir}")
        return

    print(f"📁 Images Directory Found: {images_dir}")
    print(f"📁 Train Labels Directory Found: {labels_dir} (Exists: {labels_dir.exists()})")

    # 2. Get all region folders inside the images directory
    regions = [d.name for d in images_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    print(f"🌍 Total Regions Found in 'images': {len(regions)}")

    if len(regions) == 0:
        print("⚠️ No region subfolders found inside the 'images' directory.")
        return

    # 3. Analyze regions and check corresponding labels
    annotated_count = 0
    missing_labels = []

    print("\nAnalyzing OSCD Regions:")
    for region in tqdm(regions):
        region_imgs_path = images_dir / region
        
        # Check standard bands/rect folders
        has_imgs1 = (region_imgs_path / "imgs_1_rect").exists() or (region_imgs_path / "imgs_1").exists()
        has_imgs2 = (region_imgs_path / "imgs_2_rect").exists() or (region_imgs_path / "imgs_2").exists()

        # Check if a matching mask folder/file exists in train_labels
        # Typically train_labels contains folders or files corresponding to the region name
        region_label_path = labels_dir / region
        has_label = labels_dir.exists() and (region_label_path.exists() or list(labels_dir.glob(f"*{region}*")))

        if has_label:
            annotated_count += 1
        else:
            missing_labels.append(region)

    print("\n" + "=" * 60)
    print("📊 OSCD DATASET GLOBAL SUMMARY (SPLIT STRUCTURE)")
    print("=" * 60)
    print(f"✔️ Total Regions Processed: {len(regions)}")
    print(f"🎯 Regions with Training Labels: {annotated_count}")
    print(f"🧪 Unannotated / Test Regions: {len(regions) - annotated_count}")

    if labels_dir.exists():
        print(f"🚀 Split structure parsed successfully! Ready for PyTorch/TensorFlow pipeline integration.")
    else:
        print("⚠️ 'train_labels' folder is missing. You can only run inference/unsupervised tasks.")
    print("=" * 60)

if __name__ == "__main__":
    # Point this to D:\Projects\sih_2026_ml\sih_change_detection\data\raw\oscd
    DATA_DIR = "./data/raw/oscd"
    inspect_oscd_split_dataset(DATA_DIR)