"""
YOLO model wrapper for QR code detection
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import cv2
from ultralytics import YOLO


class QRDetectionModel:
    """YOLO-based QR code detection model"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the QR detection model
        
        Args:
            model_path: Path to trained model weights. If None, uses YOLOv8 nano pretrained
        """
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"Loaded trained model from {model_path}")
        else:
            self.model = YOLO('yolo11n.pt')  # Use YOLOv11 nano as base
            print("Loaded YOLOv11 nano pretrained model")
    
    def train(self, data_yaml: str, epochs: int = 50, imgsz: int = 640, 
              batch: int = 16, project: str = 'runs', name: str = 'qr_detection',
              **kwargs) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            data_yaml: Path to YOLO data configuration file
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            project: Project directory for saving results
            name: Run name
            **kwargs: Additional training parameters
            
        Returns:
            Training results dictionary
        """
        # Set default training parameters optimized for QR detection
        train_params = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'project': project,
            'name': name,
            'patience': 100,  # Prevent early stopping for small dataset
            'exist_ok': True,
            'workers': 0,  # Set to 0 for compatibility
            **kwargs
        }
        
        print(f"Starting training with parameters: {train_params}")
        results = self.model.train(**train_params)
        
        # Update model to best weights
        best_path = os.path.join(project, name, 'weights', 'best.pt')
        if os.path.exists(best_path):
            self.model = YOLO(best_path)
            print(f"Model updated to best weights: {best_path}")
        
        return results
    
    def validate(self, data_yaml: str) -> Dict[str, float]:
        """
        Validate the model and return metrics
        
        Args:
            data_yaml: Path to YOLO data configuration file
            
        Returns:
            Dictionary containing validation metrics
        """
        metrics = self.model.val(data=data_yaml)
        
        # Calculate additional metrics
        precision = metrics.box.mp
        recall = metrics.box.mr
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            'mAP50-95': metrics.box.map,
            'mAP50': metrics.box.map50,
            'mAP75': metrics.box.map75,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
        
        print("Validation Results:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        
        return results
    
    def predict(self, source: str, conf: float = 0.25, iou: float = 0.45,
               save: bool = False, project: str = None, name: str = None) -> List[Any]:
        """
        Run inference on images
        
        Args:
            source: Path to image(s) or directory
            conf: Confidence threshold
            iou: IoU threshold for NMS
            save: Whether to save results
            project: Project directory for saving results
            name: Run name for saving results
            
        Returns:
            List of prediction results
        """
        predict_params = {
            'source': source,
            'conf': conf,
            'iou': iou,
            'save': save
        }
        
        if project:
            predict_params['project'] = project
        if name:
            predict_params['name'] = name
        
        results = self.model.predict(**predict_params)
        return results
    
    def predict_to_submission(self, images_dir: str, output_json: str,
                            conf: float = 0.25, iou: float = 0.45) -> None:
        """
        Generate submission JSON from predictions
        
        Args:
            images_dir: Directory containing test images
            output_json: Path to output JSON file
            conf: Confidence threshold
            iou: IoU threshold for NMS
        """
        results = self.predict(images_dir, conf=conf, iou=iou)
        submission_data = []
        
        for result in results:
            img_path = result.path
            img_id = os.path.basename(img_path)
            
            # Extract QR code bounding boxes
            qrs = []
            if result.boxes is not None:
                for box in result.boxes:
                    # Get bounding box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    qrs.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                    })
            
            submission_data.append({
                "image_id": img_id,
                "qrs": qrs
            })
        
        # Save to JSON
        with open(output_json, 'w') as f:
            json.dump(submission_data, f, indent=2)
        
        print(f"Submission saved to {output_json}")
        print(f"Processed {len(submission_data)} images")
        
        # Print summary statistics
        total_detections = sum(len(item['qrs']) for item in submission_data)
        print(f"Total QR codes detected: {total_detections}")
    
    def evaluate_predictions(self, predictions_json: str, ground_truth_csv: str) -> Dict[str, Any]:
        """
        Evaluate predictions against ground truth
        
        Args:
            predictions_json: Path to predictions JSON file
            ground_truth_csv: Path to ground truth CSV file
            
        Returns:
            Evaluation metrics dictionary
        """
        # Load predictions
        with open(predictions_json, 'r') as f:
            predictions = json.load(f)
        
        # Load ground truth
        gt_df = pd.read_csv(ground_truth_csv)
        gt_df.columns = ['image_id', 'x_min', 'y_min', 'x_max', 'y_max']
        gt_counts = gt_df.groupby('image_id').size()
        
        # Count predictions per image
        pred_counts = {}
        for item in predictions:
            pred_counts[item['image_id']] = len(item['qrs'])
        
        # Calculate metrics
        total_gt = gt_counts.sum()
        total_pred = sum(pred_counts.values())
        
        # Count correct predictions (simple count-based evaluation)
        correct_counts = 0
        for img_id in gt_counts.index:
            gt_count = gt_counts[img_id]
            pred_count = pred_counts.get(img_id, 0)
            correct_counts += min(gt_count, pred_count)
        
        metrics = {
            'total_ground_truth': int(total_gt),
            'total_predicted': int(total_pred),
            'correct_detections': int(correct_counts),
            'precision': correct_counts / total_pred if total_pred > 0 else 0,
            'recall': correct_counts / total_gt if total_gt > 0 else 0
        }
        
        # Calculate F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0
        
        print("Evaluation Results:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
        return metrics
    
    def save_model(self, path: str) -> None:
        """Save the model to specified path"""
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model from specified path"""
        self.model = YOLO(path)
        print(f"Model loaded from {path}")