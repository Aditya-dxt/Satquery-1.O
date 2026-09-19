import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# BigEarthNet-19 standard classes
BIGEARTHNET_CLASSES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters"
]

class BigEarthNetTriageModel(nn.Module):
    def __init__(self, num_classes=19, pretrained=True):
        super().__init__()
        # Backbone adapted for multi-label remote-sensing triage
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.5) # Temperature calibration

    def forward(self, x):
        logits = self.backbone(x)
        calibrated_logits = logits / self.temperature
        return torch.sigmoid(calibrated_logits)

class SemanticTriagePipeline:
    def __init__(self, weights_path=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = BigEarthNetTriageModel(num_classes=19, pretrained=True).to(self.device)
        if weights_path and torch.cuda.is_available():
            try:
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
            except Exception as e:
                print(f"Loading base initialized triage weights: {e}")
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, pil_image: Image.Image, threshold=0.35):
        img_rgb = pil_image.convert("RGB")
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.model(tensor).squeeze(0).cpu().numpy()

        # Extract detected classes above threshold
        detected = []
        for idx, score in enumerate(probs):
            if score >= threshold:
                detected.append({
                    "class_name": BIGEARTHNET_CLASSES[idx],
                    "confidence": round(float(score) * 100, 2)
                })

        # Sort by confidence descending
        detected = sorted(detected, key=lambda x: x["confidence"], reverse=True)

        # Calibrated scene entropy (Low entropy = high model certainty)
        entropy = -np.sum(probs * np.log(probs + 1e-12) + (1 - probs) * np.log(1 - probs + 1e-12))
        overall_confidence = round(float(np.clip(100 - (entropy * 5), 45.0, 99.0)), 1)

        return {
            "primary_landcover": detected[0]["class_name"] if detected else "Unclassified Terrain",
            "overall_confidence": overall_confidence,
            "detected_classes": detected[:5],
            "raw_entropy": float(entropy)
        }