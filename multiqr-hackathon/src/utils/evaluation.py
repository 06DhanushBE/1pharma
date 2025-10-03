"""
Utility functions for QR code detection project
"""
import os
import json
import cv2
import pandas as pd
from typing import List, Dict, Any, Tuple
import numpy as np


def visualize_predictions(image_path: str, predictions: List[Dict], 
                         ground_truth: List[Dict] = None, 
                         output_path: str = None) -> np.ndarray:
    """
    Visualize predictions and optionally ground truth on an image
    
    Args:
        image_path: Path to the input image
        predictions: List of prediction dictionaries with 'bbox' key
        ground_truth: Optional list of ground truth dictionaries with bbox
        output_path: Optional path to save the visualization
        
    Returns:
        Annotated image as numpy array
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Draw predictions in green
    for pred in predictions:
        bbox = pred['bbox']
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, 'PRED', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw ground truth in red if provided
    if ground_truth:
        for gt in ground_truth:
            bbox = gt['bbox']
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, 'GT', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Visualization saved to {output_path}")
    
    return img


def load_ground_truth_for_image(csv_path: str, image_id: str) -> List[Dict]:
    """
    Load ground truth bounding boxes for a specific image
    
    Args:
        csv_path: Path to ground truth CSV file
        image_id: Image identifier
        
    Returns:
        List of ground truth bounding boxes
    """
    df = pd.read_csv(csv_path)
    df.columns = ['image_id', 'x_min', 'y_min', 'x_max', 'y_max']
    
    image_data = df[df['image_id'] == image_id]
    ground_truth = []
    
    for _, row in image_data.iterrows():
        ground_truth.append({
            'bbox': [int(row['x_min']), int(row['y_min']), 
                    int(row['x_max']), int(row['y_max'])]
        })
    
    return ground_truth


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def evaluate_detection_precision_recall(predictions_json: str, 
                                       ground_truth_csv: str,
                                       iou_threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate detection performance using precision and recall with IoU matching
    
    Args:
        predictions_json: Path to predictions JSON file
        ground_truth_csv: Path to ground truth CSV file
        iou_threshold: IoU threshold for considering a detection as correct
        
    Returns:
        Dictionary with precision, recall, and F1 score
    """
    # Load predictions
    with open(predictions_json, 'r') as f:
        predictions = json.load(f)
    
    # Load ground truth
    gt_df = pd.read_csv(ground_truth_csv)
    gt_df.columns = ['image_id', 'x_min', 'y_min', 'x_max', 'y_max']
    
    total_predictions = 0
    total_ground_truth = 0
    true_positives = 0
    
    for pred_item in predictions:
        image_id = pred_item['image_id']
        pred_boxes = [qr['bbox'] for qr in pred_item['qrs']]
        
        # Get ground truth for this image
        gt_boxes = []
        image_gt = gt_df[gt_df['image_id'] == image_id]
        for _, row in image_gt.iterrows():
            gt_boxes.append([int(row['x_min']), int(row['y_min']), 
                           int(row['x_max']), int(row['y_max'])])
        
        total_predictions += len(pred_boxes)
        total_ground_truth += len(gt_boxes)
        
        # Match predictions to ground truth using IoU
        matched_gt = set()
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt_box in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_iou >= iou_threshold:
                true_positives += 1
                matched_gt.add(best_gt_idx)
    
    # Calculate metrics
    precision = true_positives / total_predictions if total_predictions > 0 else 0
    recall = true_positives / total_ground_truth if total_ground_truth > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'total_predictions': total_predictions,
        'total_ground_truth': total_ground_truth
    }
    
    return metrics


def create_sample_submission(images_dir: str, output_json: str, 
                           num_samples: int = 5) -> None:
    """
    Create a sample submission file with dummy predictions
    
    Args:
        images_dir: Directory containing images
        output_json: Path to output JSON file
        num_samples: Number of sample images to include
    """
    image_files = [f for f in os.listdir(images_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) > num_samples:
        image_files = image_files[:num_samples]
    
    submission_data = []
    
    for img_file in image_files:
        # Create dummy predictions (empty for now)
        submission_data.append({
            "image_id": img_file,
            "qrs": []  # No predictions in sample
        })
    
    with open(output_json, 'w') as f:
        json.dump(submission_data, f, indent=2)
    
    print(f"Sample submission created: {output_json}")


def validate_submission_format(submission_json: str) -> bool:
    """
    Validate that submission JSON follows the required format
    
    Args:
        submission_json: Path to submission JSON file
        
    Returns:
        True if format is valid, False otherwise
    """
    try:
        with open(submission_json, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("Error: Submission must be a list")
            return False
        
        for item in data:
            if not isinstance(item, dict):
                print("Error: Each item must be a dictionary")
                return False
            
            if 'image_id' not in item or 'qrs' not in item:
                print("Error: Each item must have 'image_id' and 'qrs' keys")
                return False
            
            if not isinstance(item['qrs'], list):
                print("Error: 'qrs' must be a list")
                return False
            
            for qr in item['qrs']:
                if not isinstance(qr, dict) or 'bbox' not in qr:
                    print("Error: Each QR must be a dict with 'bbox' key")
                    return False
                
                bbox = qr['bbox']
                if not isinstance(bbox, list) or len(bbox) != 4:
                    print("Error: bbox must be a list of 4 numbers")
                    return False
                
                if not all(isinstance(x, (int, float)) for x in bbox):
                    print("Error: bbox coordinates must be numbers")
                    return False
        
        print("Submission format is valid")
        return True
        
    except Exception as e:
        print(f"Error validating submission: {e}")
        return False


def copy_demo_images(source_dir: str, dest_dir: str, num_images: int = 10) -> None:
    """
    Copy a few demo images for testing
    
    Args:
        source_dir: Source directory containing images
        dest_dir: Destination directory
        num_images: Number of images to copy
    """
    os.makedirs(dest_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(source_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) > num_images:
        image_files = image_files[:num_images]
    
    for img_file in image_files:
        src_path = os.path.join(source_dir, img_file)
        dst_path = os.path.join(dest_dir, img_file)
        
        img = cv2.imread(src_path)
        if img is not None:
            cv2.imwrite(dst_path, img)
            print(f"Copied {img_file} to demo directory")
    
    print(f"Copied {len(image_files)} demo images to {dest_dir}")