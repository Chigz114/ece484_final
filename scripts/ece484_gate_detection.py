"""
Gate Detection Script

This script reads an image, applies gate detection using deep learning,
and displays the original image along with the generated mask.

"""
import numpy as np
import cv2
import os
import sys
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add parent directory to path to import UNet
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(script_dir)
train_gate_dir = os.path.join(parent_dir, 'train_gate')
sys.path.insert(0, train_gate_dir)

from train_gate_detection import UNet

# Global model instance (loaded once)
_model = None
_device = None
_transform = None

def _load_model():
    """Load the trained UNet model (only once)."""
    global _model, _device, _transform
    
    if _model is not None:
        return _model, _device, _transform
    
    # Determine device
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model (use the latest trained model)
    model_path = os.path.join(parent_dir, 'train_gate', 'trained_models_new_gates_v2', 'best_model.pth')
    _model = UNet(in_channels=3, out_channels=1).to(_device)
    
    checkpoint = torch.load(model_path, map_location=_device, weights_only=False)
    _model.load_state_dict(checkpoint['model_state_dict'])
    _model.eval()
    
    # Define transform
    _transform = A.Compose([
        A.Resize(480, 640),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    print(f"✓ Gate detection model loaded (device: {_device})")
    
    return _model, _device, _transform


def gate_detection(image, debug=False):
    """
    Detects gates in an image using deep learning (UNet).

    Args:
        image: The input BGR image (numpy array).
        debug: If True, displays intermediate visualization steps.

    Returns:
        A binary mask of the detected gates (numpy array, 0-255).
    """
    # Load model (only once)
    model, device, transform = _load_model()
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    transformed = transform(image=image_rgb)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        mask_pred = model(image_tensor)
        mask_pred = torch.sigmoid(mask_pred)
        mask_np = (mask_pred[0, 0].cpu().numpy() * 255).astype(np.uint8)
    
    # Threshold to binary
    _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
    
    if debug:
        cv2.imshow("Predicted Mask", mask_np)
        cv2.imshow("Binary Mask", binary_mask)
    
    return binary_mask



if __name__ == "__main__":
    # Use sample image from ns-renderer.py to check your gate detection algorithm.
    script_dir = os.path.dirname(os.path.realpath(__file__))
    image_path = os.path.join(script_dir, 'gate-detect-Yan-example', '0050.png')
    # image_path = './images/combo.png' # Using the image from generate_image.py
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image at {image_path}")
    else:
        # Run detection with debug visualization enabled
        mask = gate_detection(image, debug=True)

        # Create a blended view
        blended_image = cv2.addWeighted(image, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)

        cv2.imshow('Original Image', image)
        cv2.imshow('Final Mask', mask)
        cv2.imshow('Blended View', blended_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
