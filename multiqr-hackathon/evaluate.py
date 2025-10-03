#!/usr/bin/env python3
"""
Evaluation Script for QR Detection
Evaluates model predictions against ground truth annotations

Usage:
    python evaluate.py --predictions outputs/submission_detection_1.json --ground-truth combined.csv
    python evaluate.py --predictions outputs/submission_detection_1.json --ground-truth combined.csv --visualize
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.evaluation import (
    evaluate_detection_precision_recall, 
    visualize_predictions,
    load_ground_truth_for_image,
    validate_submission_format
)


def create_detailed_report(predictions_json, ground_truth_csv, output_dir):
    """Create detailed evaluation report with per-image analysis"""
    
    # Load data
    with open(predictions_json, 'r') as f:
        predictions = json.load(f)
    
    import pandas as pd
    gt_df = pd.read_csv(ground_truth_csv)
    gt_df.columns = ['image_id', 'x_min', 'y_min', 'x_max', 'y_max']
    
    # Per-image analysis
    report_data = []
    
    for pred_item in predictions:
        image_id = pred_item['image_id']
        pred_count = len(pred_item['qrs'])
        
        # Get ground truth count
        gt_count = len(gt_df[gt_df['image_id'] == image_id])
        
        report_data.append({
            'image_id': image_id,
            'ground_truth_qrs': gt_count,
            'predicted_qrs': pred_count,
            'difference': pred_count - gt_count,
            'accuracy': min(pred_count, gt_count) / max(pred_count, gt_count, 1)
        })
    
    # Save detailed report
    report_df = pd.DataFrame(report_data)
    report_path = os.path.join(output_dir, 'detailed_evaluation_report.csv')
    report_df.to_csv(report_path, index=False)
    
    # Print summary statistics
    print(f"\nDetailed Report saved to: {report_path}")
    print(f"Images with correct count: {len(report_df[report_df['difference'] == 0])}")
    print(f"Images over-detected: {len(report_df[report_df['difference'] > 0])}")
    print(f"Images under-detected: {len(report_df[report_df['difference'] < 0])}")
    print(f"Average accuracy per image: {report_df['accuracy'].mean():.4f}")
    
    return report_df


def main():
    parser = argparse.ArgumentParser(description='Evaluate QR Detection Results')
    
    # Input arguments
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions JSON file')
    parser.add_argument('--ground-truth', type=str, required=True,
                       help='Path to ground truth CSV file')
    parser.add_argument('--images-dir', type=str,
                       help='Path to images directory for visualization')
    
    # Evaluation arguments
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for matching predictions to ground truth')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='outputs/evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization images')
    parser.add_argument('--detailed-report', action='store_true',
                       help='Create detailed per-image evaluation report')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.predictions):
        print(f"Error: Predictions file not found: {args.predictions}")
        sys.exit(1)
    
    if not os.path.exists(args.ground_truth):
        print(f"Error: Ground truth file not found: {args.ground_truth}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Validate submission format
        print("Validating submission format...")
        if not validate_submission_format(args.predictions):
            print("Submission format validation failed!")
            sys.exit(1)
        
        # Run IoU-based evaluation
        print(f"Running evaluation with IoU threshold: {args.iou_threshold}")
        metrics = evaluate_detection_precision_recall(
            args.predictions, 
            args.ground_truth, 
            args.iou_threshold
        )
        
        # Print main metrics
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"True Positives: {metrics['true_positives']}")
        print(f"Total Predictions: {metrics['total_predictions']}")
        print(f"Total Ground Truth: {metrics['total_ground_truth']}")
        
        # Save metrics to JSON
        metrics_path = os.path.join(args.output_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")
        
        # Create detailed report if requested
        if args.detailed_report:
            print("\nCreating detailed evaluation report...")
            report_df = create_detailed_report(
                args.predictions, 
                args.ground_truth, 
                args.output_dir
            )
        
        # Create visualizations if requested
        if args.visualize and args.images_dir:
            if not os.path.exists(args.images_dir):
                print(f"Warning: Images directory not found: {args.images_dir}")
            else:
                print(f"\nCreating visualization images...")
                viz_dir = os.path.join(args.output_dir, 'visualizations')
                os.makedirs(viz_dir, exist_ok=True)
                
                # Load predictions
                with open(args.predictions, 'r') as f:
                    predictions = json.load(f)
                
                viz_count = 0
                for item in predictions[:20]:  # Visualize first 20 images
                    image_id = item['image_id']
                    image_path = os.path.join(args.images_dir, image_id)
                    
                    if os.path.exists(image_path):
                        try:
                            # Get ground truth for this image
                            ground_truth = load_ground_truth_for_image(args.ground_truth, image_id)
                            
                            # Create visualization
                            output_viz_path = os.path.join(viz_dir, f"eval_{image_id}")
                            visualize_predictions(
                                image_path=image_path,
                                predictions=item['qrs'],
                                ground_truth=ground_truth,
                                output_path=output_viz_path
                            )
                            viz_count += 1
                        except Exception as e:
                            print(f"Warning: Could not create visualization for {image_id}: {e}")
                
                print(f"Created {viz_count} visualization images in {viz_dir}")
        
        print("\n" + "="*50)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()