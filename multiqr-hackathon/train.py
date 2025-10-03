#!/usr/bin/env python3
"""
Multi-QR Code Detection Training Script
Trains YOLOv11 model for QR code detection on medicine packs

Usage:
    python train.py --data data/QR_Dataset --annotations combined.csv --epochs 50
    
Note: This script is designed to work both locally and on Kaggle.
For Kaggle training, use the provided notebook version.
"""

import argparse
import os
import sys
import shutil
from pathlib import Path

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.datasets.preprocessing import combine_annotations, create_yolo_dataset
from src.models.yolo_detector import QRDetectionModel


def setup_kaggle_paths(args):
    """Setup paths for Kaggle environment"""
    if args.kaggle:
        # Kaggle-specific paths
        args.data_dir = '/kaggle/input/qrcodes/QR_Dataset'
        args.annotations = '/kaggle/input/annoted/combined.csv'
        args.output_dir = '/kaggle/working/datasets'
        args.project_dir = '/kaggle/working/yolo_runs'
        print("Using Kaggle paths")
    else:
        # Local paths
        if not args.data_dir:
            args.data_dir = 'data/QR_Dataset'
        if not args.output_dir:
            args.output_dir = 'datasets'
        if not args.project_dir:
            args.project_dir = 'runs'
        print("Using local paths")


def prepare_dataset(args):
    """Prepare YOLO dataset from images and annotations"""
    print("Preparing YOLO dataset...")
    
    # Create combined images directory
    train_images_dir = os.path.join(args.data_dir, 'train_images')
    test_images_dir = os.path.join(args.data_dir, 'test_images')
    combined_images_dir = os.path.join(args.output_dir, 'combined_images')
    
    os.makedirs(combined_images_dir, exist_ok=True)
    
    # Copy all images to combined directory
    if os.path.exists(train_images_dir):
        for img_file in os.listdir(train_images_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src = os.path.join(train_images_dir, img_file)
                dst = os.path.join(combined_images_dir, img_file)
                shutil.copy2(src, dst)
    
    if os.path.exists(test_images_dir):
        for img_file in os.listdir(test_images_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src = os.path.join(test_images_dir, img_file)
                dst = os.path.join(combined_images_dir, img_file)
                shutil.copy2(src, dst)
    
    print(f"Images combined in {combined_images_dir}")
    
    # Create YOLO dataset structure
    yolo_dataset_dir = os.path.join(args.output_dir, 'yolo_dataset')
    create_yolo_dataset(
        images_dir=combined_images_dir,
        annotations_csv=args.annotations,
        output_dir=yolo_dataset_dir,
        train_range=(1, 200),
        test_range=(201, 250)
    )
    
    # Return path to data.yaml
    return os.path.join(yolo_dataset_dir, 'data.yaml')


def train_model(args, data_yaml_path):
    """Train the QR detection model"""
    print("Starting model training...")
    
    # Initialize model
    model = QRDetectionModel()
    
    # Training parameters
    train_params = {
        'data': data_yaml_path,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'project': args.project_dir,
        'name': args.name,
        'patience': 100,
        'exist_ok': True,
        'workers': 0 if args.kaggle else 4,
        'device': 'cuda' if args.gpu else 'cpu'
    }
    
    # Train the model
    results = model.train(**train_params)
    
    # Validate the model
    print("\nValidating model...")
    metrics = model.validate(data_yaml_path)
    
    # Save best model to outputs directory
    best_model_path = os.path.join(args.project_dir, args.name, 'weights', 'best.pt')
    output_model_path = os.path.join('outputs', 'best.pt')
    
    os.makedirs('outputs', exist_ok=True)
    if os.path.exists(best_model_path):
        shutil.copy2(best_model_path, output_model_path)
        print(f"Best model saved to {output_model_path}")
    
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description='Train QR Detection Model')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, help='Path to QR_Dataset directory')
    parser.add_argument('--annotations', type=str, default='combined.csv',
                       help='Path to combined annotations CSV file')
    parser.add_argument('--output-dir', type=str, help='Output directory for datasets')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU for training')
    
    # Output arguments
    parser.add_argument('--project-dir', type=str,
                       help='Project directory for saving results')
    parser.add_argument('--name', type=str, default='qr_detection',
                       help='Run name')
    
    # Environment arguments
    parser.add_argument('--kaggle', action='store_true',
                       help='Use Kaggle environment paths')
    
    args = parser.parse_args()
    
    # Setup paths based on environment
    setup_kaggle_paths(args)
    
    try:
        # Prepare dataset
        data_yaml_path = prepare_dataset(args)
        print(f"Dataset prepared with config: {data_yaml_path}")
        
        # Train model
        model, metrics = train_model(args, data_yaml_path)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Best model available at: outputs/best.pt")
        print(f"Training results in: {args.project_dir}/{args.name}")
        
        # Print final metrics
        print("\nFinal Validation Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()