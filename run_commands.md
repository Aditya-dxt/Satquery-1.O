# ==========================================
# SIH1518: Change Detection Pipeline Script
# ==========================================

# 1. Initialize project folder structure (if not already done)
python setup_project.py

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create a validation split from the training set (since LEVIR-CD only has train/test)
python src/create_val_levir_cd.py

# 4. Inspect dataset integrity, shape consistency, and class imbalance ratios
python src/levir_inspection.py
python src/oscd_inspection.py

# 5. Train the GPU-optimized Siamese UNet model
python src/train.py --data_dir ./data/raw/levir_cd --epochs 30 --batch_size 16 --lr 0.001

# 6. Run inference script to generate change prediction maps
python src/inference.py --img_a ./data/raw/levir_cd/test/A/00001.png --img_b ./data/raw/levir_cd/test/B/00001.png --weights ./models/best_siamese_model.pth

python src/evaluate.py --data_dir ./data/raw/levir_cd --weights ./models/best_siamese_model.pth

# 7. Launch the interactive presentation dashboard
streamlit run dashboard/app.py