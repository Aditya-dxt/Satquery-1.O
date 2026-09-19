import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class SemanticChangeClassifier:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initializes the Zero-Shot CLIP model for satellite image patch classification.
        """
        print("📦 Loading Zero-Shot CLIP Semantic Classifier...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Define rich text prompts tailored for satellite change detection (SIH 1518)
        self.prompts = [
            "an aerial satellite photo of illegal deforestation or tree clearing",
            "an aerial satellite photo of new urban building construction",
            "an aerial satellite photo of open-cast mining or soil excavation",
            "an aerial satellite photo of agricultural land expansion",
            "an aerial satellite photo of unchanged natural terrain or normal land"
        ]
        
        # Clean labels to display on your dashboard UI and compliance report
        self.class_names = [
            "Deforestation / Tree Clearing",
            "Urban Building Construction",
            "Open-Cast Mining / Excavation",
            "Agricultural Expansion",
            "Unaligned / Natural Background"
        ]

    @torch.no_grad()
    def classify_patch(self, patch_path: str):
        """
        Classifies a single cropped patch image against the defined human activity prompts.
        """
        image = Image.open(patch_path).convert("RGB")
        
        inputs = self.processor(
            text=self.prompts, 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # Image-text similarity scores
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        
        best_idx = int(probs.argmax())
        confidence = float(probs[best_idx]) * 100
        
        return {
            "classification": self.class_names[best_idx],
            "confidence": round(confidence, 2),
            "all_probabilities": {
                self.class_names[i]: round(float(probs[i]) * 100, 2) 
                for i in range(len(self.class_names))
            }
        }