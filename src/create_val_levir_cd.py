import os, random, shutil
base_train_A = './data/raw/levir_cd/train/A'
base_train_B = './data/raw/levir_cd/train/B'
base_train_lab = './data/raw/levir_cd/train/label'
val_A = './data/raw/levir_cd/val/A'
val_B = './data/raw/levir_cd/val/B'
val_lab = './data/raw/levir_cd/val/label'
os.makedirs(val_A, exist_ok=True); os.makedirs(val_B, exist_ok=True); os.makedirs(val_lab, exist_ok=True)
images = sorted(os.listdir(base_train_A))
random.seed(42); random.shuffle(images)
val_size = int(len(images) * 0.15)
for img_name in images[:val_size]:
    shutil.move(os.path.join(base_train_A, img_name), os.path.join(val_A, img_name))
    shutil.move(os.path.join(base_train_B, img_name), os.path.join(val_B, img_name))
    shutil.move(os.path.join(base_train_lab, img_name), os.path.join(val_lab, img_name))
print('Validation split created successfully!')