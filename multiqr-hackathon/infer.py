#!/usr/bin/env python3
"""
Multi-QR Code Detection Inference Script
Generates submission JSON for QR code detection on test images

Usage:
    python infer.py --input data/demo_images --output outputs/submission_detection_1.json
    python infer.py --input data/test_images --model outputs/best.pt --conf 0.25
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.yolo_detector import QRDetectionModel
from src.utils.evaluation import validate_submission_format, visualize_predictions


def main():
    parser = argparse.ArgumentParser(description='QR Detection Inference')
    
    # Input/Output arguments
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input images directory')
    parser.add_argument('--output', type=str, default='outputs/submission_detection_1.json',
                       help='Path to output submission JSON file')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='outputs/best.pt',
                       help='Path to trained model weights')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for detection')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    
    # Visualization arguments
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualization images')
    parser.add_argument('--viz-dir', type=str, default='outputs/visualizations',
                       help='Directory to save visualization images')
    
    # Evaluation arguments
    parser.add_argument('--ground-truth', type=str,
                       help='Path to ground truth CSV for evaluation')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please train the model first using: python train.py")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        # Load the trained model
        print(f"Loading model from {args.model}...")
        model = QRDetectionModel(args.model)
        
        # Run inference and generate submission
        print(f"Running inference on images in {args.input}...")
        model.predict_to_submission(
            images_dir=args.input,
            output_json=args.output,
            conf=args.conf,
            iou=args.iou
        )
        
        # Validate submission format
        print("Validating submission format...")
        if validate_submission_format(args.output):
            print(f"✓ Submission saved successfully: {args.output}")
        else:
            print("✗ Submission format validation failed!")
            sys.exit(1)
        
        # Load and print submission summary
        with open(args.output, 'r') as f:
            submission_data = json.load(f)
        
        total_images = len(submission_data)
        total_detections = sum(len(item['qrs']) for item in submission_data)
        images_with_detections = sum(1 for item in submission_data if len(item['qrs']) > 0)
        
        print("\n" + "="*50)
        print("INFERENCE SUMMARY")
        print("="*50)
        print(f"Total images processed: {total_images}")
        print(f"Images with QR detections: {images_with_detections}")
        print(f"Total QR codes detected: {total_detections}")
        print(f"Average QR codes per image: {total_detections/total_images:.2f}")
        
        # Create visualizations if requested
        if args.visualize:
            print(f"\nCreating visualizations in {args.viz_dir}...")
            os.makedirs(args.viz_dir, exist_ok=True)
            
            viz_count = 0
            for item in submission_data[:10]:  # Visualize first 10 images
                image_id = item['image_id']
                image_path = os.path.join(args.input, image_id)
                
                if os.path.exists(image_path):
                    predictions = item['qrs']
                    output_viz_path = os.path.join(args.viz_dir, f"viz_{image_id}")
                    
                    try:
                        visualize_predictions(
                            image_path=image_path,
                            predictions=predictions,
                            output_path=output_viz_path
                        )
                        viz_count += 1
                    except Exception as e:
                        print(f"Warning: Could not create visualization for {image_id}: {e}")
            
            print(f"Created {viz_count} visualization images")
        
        # Evaluate against ground truth if provided
        if args.ground_truth and os.path.exists(args.ground_truth):
            print(f"\nEvaluating against ground truth: {args.ground_truth}")
            metrics = model.evaluate_predictions(args.output, args.ground_truth)
            
            print("\nEvaluation Results:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
        
        print("\n" + "="*50)
        print("INFERENCE COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Submission file: {args.output}")
        if args.visualize:
            print(f"Visualizations: {args.viz_dir}")
        
    except Exception as e:
        print(f"Error during inference: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()