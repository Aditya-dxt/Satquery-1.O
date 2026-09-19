import cv2
import numpy as np
import os
from .classifier import SemanticChangeClassifier

# Initialize classifier once globally so it doesn't reload on every image request
classifier = SemanticChangeClassifier()

def process_detected_changes(mask_path, post_image_path, output_dir, min_area=15, m2_per_pixel=100):
    """
    Processes binary change mask, crops post-image patches, calculates area,
    runs semantic classification, assigns severity, and outputs metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    patches_dir = os.path.join(output_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    post_img = cv2.imread(post_image_path)
    
    if mask is None or post_img is None:
        raise ValueError("Failed to load mask or post-change image.")
        
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    change_events = []
    annotated_img = post_img.copy()
    
    for idx, contour in enumerate(contours):
        area_pixels = cv2.contourArea(contour)
        if area_pixels < min_area:
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        patch = post_img[y:y+h, x:x+w]
        
        area_sq_meters = float(area_pixels * m2_per_pixel)
        
        # Save cropped patch image
        patch_filename = f"event_{idx+1}.png"
        patch_save_path = os.path.join(patches_dir, patch_filename)
        cv2.imwrite(patch_save_path, patch)
        
        # --- Run Semantic Classification via CLIP ---
        classification_result = classifier.classify_patch(patch_save_path)
        activity_type = classification_result["classification"]
        confidence = classification_result["confidence"]
        
        # Determine Severity Level based on area and activity type
        if area_sq_meters > 5000 or "Deforestation" in activity_type:
            severity = "HIGH"
            color = (0, 0, 255)      # Red (BGR)
        elif area_sq_meters > 1000:
            severity = "MEDIUM"
            color = (0, 165, 255)    # Orange (BGR)
        else:
            severity = "LOW"
            color = (0, 255, 255)    # Yellow (BGR)
            
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated_img, f"#{idx+1} {activity_type[:10]}...", (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        change_events.append({
            "id": idx + 1,
            "bbox": [int(x), int(y), int(w), int(h)],
            "pixel_area": int(area_pixels),
            "area_sq_m": area_sq_meters,
            "activity_type": activity_type,
            "confidence": confidence,
            "severity": severity,
            "patch_url": f"/static/uploads/patches/{patch_filename}"
        })
        
    annotated_save_path = os.path.join(output_dir, "annotated_overview.png")
    cv2.imwrite(annotated_save_path, annotated_img)
    
    return change_events, "static/uploads/annotated_overview.png"