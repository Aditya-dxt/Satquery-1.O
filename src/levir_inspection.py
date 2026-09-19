import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def inspect_dataset(data_dir="./data/raw/levir_cd"):
    print("=" * 60)
    print(f"🔎 STARTING DATASET INSPECTION FOR: {data_dir}")
    print("=" * 60)
    
    splits = [d for d in ['train', 'val', 'test'] if os.path.exists(os.path.join(data_dir, d))]
    
    if not splits:
        print(f"❌ Error: No valid splits ('train', 'val', 'test') found in {data_dir}!")
        print("Make sure you have unzipped/placed your dataset correctly.")
        return

    total_global_pixels = 0
    total_changed_pixels = 0

    for split in splits:
        print(f"\n📂 Inspecting Split: [{split.upper()}]")
        
        dir_A = os.path.join(data_dir, split, 'A')
        dir_B = os.path.join(data_dir, split, 'B')
        dir_label = os.path.join(data_dir, split, 'label')
        
        # Verify subfolders exist
        if not all(os.path.exists(p) for p in [dir_A, dir_B, dir_label]):
            print(f"   ❌ Missing structural subfolders ('A', 'B', or 'label') inside '{split}'!")
            continue
            
        files_A = sorted(os.listdir(dir_A))
        files_B = sorted(os.listdir(dir_B))
        files_label = sorted(os.listdir(dir_label))
        
        print(f"   - Total Image Pairs Found: {len(files_A)}")
        
        # 1. Integrity Check: File counts match
        if not (len(files_A) == len(files_B) == len(files_label)):
            print("   ⚠️ WARNING: Mismatch in number of files between A, B, and labels!")
            
        # 2. Integrity Check: File names match exactly
        if files_A != files_B or files_A != files_label:
            print("   ⚠️ WARNING: Filenames across A, B, and label directories do not match completely!")

        split_pixels = 0
        split_changed = 0
        resolutions = set()
        corrupted_files = 0

        for fname in tqdm(files_A, desc=f"   Analyzing {split} files"):
            path_A = os.path.join(dir_A, fname)
            path_B = os.path.join(dir_B, fname)
            path_lab = os.path.join(dir_label, fname)
            
            try:
                # Open images to verify integrity and layout
                img_A = Image.open(path_A)
                img_B = Image.open(path_B)
                lab = Image.open(path_lab)
                
                resolutions.add(img_A.size)
                
                # Check dimensions match across pair and mask
                if img_A.size != img_B.size or img_A.size != lab.size:
                    print(f"\n   ⚠️ Size mismatch in file: {fname}")
                
                # Calculate class distribution from mask
                lab_arr = np.array(lab)
                # Standardize mask values (supports 0-255 or 0-1)
                if lab_arr.max() > 1:
                    lab_arr = lab_arr / 255.0
                
                flat_lab = lab_arr.flatten()
                split_pixels += flat_lab.size
                split_changed += np.sum(flat_lab > 0.5)
                
            except Exception as e:
                corrupted_files += 1
                print(f"\n   ❌ Corrupted or unreadable file: {fname} | Error: {e}")

        total_global_pixels += split_pixels
        total_changed_pixels += split_changed
        
        # Split metrics summary
        print(f"   - Unique Image Resolutions: {resolutions}")
        print(f"   - Corrupted Files Found: {corrupted_files}")
        if split_pixels > 0:
            change_ratio = (split_changed / split_pixels) * 100
            print(f"   - Human Activity / Change Pixel Ratio: {change_ratio:.2f}%")

    print("\n" + "=" * 60)
    print("📊 GLOBAL SUMMARY ACROSS DATASET")
    print("=" * 60)
    if total_global_pixels > 0:
        global_change_ratio = (total_changed_pixels / total_global_pixels) * 100
        print(f"🌐 Overall Change/Construction Coverage: {global_change_ratio:.2f}%")
        print(f"💡 AI Insight: Changes occupy less than {global_change_ratio:.1f}% of pixels.")
        print("   -> Action confirmed: Your Hybrid Loss (Focal + Dice) script is mandatory to prevent zero-change shortcutting!")
    print("=" * 60)

if __name__ == "__main__":
    inspect_dataset()