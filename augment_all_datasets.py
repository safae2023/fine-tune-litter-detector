#!/usr/bin/env python3
"""
Comprehensive Dataset Augmentation Script
Processes all images in all batch folders and generates new annotation files
with transformed bounding boxes for all augmented images.
"""

import cv2
import albumentations as A
from pathlib import Path
import json
import time
import os
import copy

class DatasetAugmenter:
    def __init__(self, data_path="data", output_path="augmented_datasets"):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        
        # Create main output directory
        self.output_path.mkdir(exist_ok=True)
        
        # Load original annotations
        self.original_annotations = self.load_annotations()
        
        # New annotations structure for augmented dataset
        self.augmented_annotations = {
            "info": self.original_annotations.get("info", {}),
            "licenses": self.original_annotations.get("licenses", []),
            "categories": self.original_annotations.get("categories", []),
            "images": [],
            "annotations": []
        }
        
        # Define augmentation pipeline
        self.transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))
        
        # Track annotation IDs
        self.next_annotation_id = 1
        self.next_image_id = 1
    
    def load_annotations(self):
        """Load COCO format annotations"""
        annotations_path = self.data_path / "annotations.json"
        try:
            with open(annotations_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {annotations_path} not found.")
            return {"images": [], "annotations": [], "categories": []}
    
    def get_image_annotations(self, image_name):
        """Get annotations for a specific image"""
        image_id = None
        for img in self.original_annotations["images"]:
            img_filename = img["file_name"]
            if "/" in img_filename:
                img_filename = img_filename.split("/")[-1]
            
            if img_filename == image_name:
                image_id = img["id"]
                break
        
        if image_id is None:
            return [], []
        
        bboxes = []
        category_ids = []
        for ann in self.original_annotations["annotations"]:
            if ann["image_id"] == image_id:
                bboxes.append(ann["bbox"])
                category_ids.append(ann["category_id"])
        
        return bboxes, category_ids
    
    def draw_bboxes(self, image, bboxes, category_ids):
        """Draw bounding boxes on image"""
        img_with_boxes = image.copy()
        
        for bbox, cat_id in zip(bboxes, category_ids):
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            
            # Ensure coordinates are within image bounds
            x = max(0, min(x, image.shape[1] - 1))
            y = max(0, min(y, image.shape[0] - 1))
            w = max(1, min(w, image.shape[1] - x))
            h = max(1, min(h, image.shape[0] - y))
            
            # Draw rectangle and label
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 3)
            label = f"ID:{cat_id}"
            cv2.putText(img_with_boxes, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return img_with_boxes
    
    def add_image_to_annotations(self, image_path, batch_folder, is_augmented=False, aug_index=None):
        """Add image entry to annotations"""
        # Get original image info
        original_filename = image_path.name
        original_image_id = None
        
        for img in self.original_annotations["images"]:
            if img["file_name"].endswith(original_filename):
                original_image_id = img["id"]
                break
        
        if original_image_id is None:
            return None
        
        # Find original image info
        original_img_info = None
        for img in self.original_annotations["images"]:
            if img["id"] == original_image_id:
                original_img_info = img
                break
        
        if original_img_info is None:
            return None
        
        # Create new image entry
        new_image_info = copy.deepcopy(original_img_info)
        new_image_info["id"] = self.next_image_id
        
        if is_augmented:
            new_image_info["file_name"] = f"{batch_folder}/{image_path.stem}_aug_{aug_index}.jpg"
        else:
            new_image_info["file_name"] = f"{batch_folder}/{image_path.stem}_original.jpg"
        
        # Add to annotations
        self.augmented_annotations["images"].append(new_image_info)
        
        return self.next_image_id
    
    def add_annotations_for_image(self, image_id, bboxes, category_ids, is_augmented=False):
        """Add annotation entries for an image"""
        for bbox, cat_id in zip(bboxes, category_ids):
            annotation = {
                "id": self.next_annotation_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],  # width * height
                "iscrowd": 0
            }
            
            self.augmented_annotations["annotations"].append(annotation)
            self.next_annotation_id += 1
    
    def process_single_image(self, image_path, output_folder, batch_folder, num_augmentations=2):
        """Process a single image with augmentations and update annotations"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return False
        
        # Get annotations
        bboxes, category_ids = self.get_image_annotations(image_path.name)
        
        if not bboxes:
            return False
        
        # Save original image with bounding boxes
        original_with_boxes = self.draw_bboxes(image, bboxes, category_ids)
        original_output = output_folder / f"{image_path.stem}_original.jpg"
        cv2.imwrite(str(original_output), original_with_boxes)
        
        # Add original image to annotations
        original_image_id = self.add_image_to_annotations(image_path, batch_folder, is_augmented=False)
        if original_image_id:
            self.add_annotations_for_image(original_image_id, bboxes, category_ids, is_augmented=False)
        
        # Apply augmentations
        for i in range(num_augmentations):
            transformed = self.transform(image=image, bboxes=bboxes, category_ids=category_ids)
            aug_img = transformed["image"]
            aug_bboxes = transformed["bboxes"]
            aug_cats = transformed["category_ids"]
            
            # Draw bounding boxes on augmented image
            aug_with_boxes = self.draw_bboxes(aug_img, aug_bboxes, aug_cats)
            
            # Save augmented image
            output_name = f"{image_path.stem}_aug_{i+1}.jpg"
            output_path = output_folder / output_name
            cv2.imwrite(str(output_path), aug_with_boxes)
            
            # Add augmented image to annotations
            aug_image_id = self.add_image_to_annotations(image_path, batch_folder, is_augmented=True, aug_index=i+1)
            if aug_image_id:
                self.add_annotations_for_image(aug_image_id, aug_bboxes, aug_cats, is_augmented=True)
        
        return True
    
    def process_all_datasets(self, num_augmentations=2):
        """Process all images in all batch folders"""
        print("Starting comprehensive dataset augmentation...")
        print("=" * 60)
        
        # Find all batch folders
        batch_folders = []
        for item in self.data_path.iterdir():
            if item.is_dir() and item.name.startswith("batch_"):
                batch_folders.append(item.name)
        
        batch_folders.sort()  # Sort to process in order
        print(f"Found {len(batch_folders)} batch folders: {batch_folders}")
        
        # Process each batch
        total_images = 0
        total_processed = 0
        start_time = time.time()
        
        for batch_folder in batch_folders:
            batch_path = self.data_path / batch_folder
            output_batch_path = self.output_path / batch_folder
            
            # Create output folder for this batch
            output_batch_path.mkdir(exist_ok=True)
            
            # Get all images in this batch
            images = list(batch_path.glob("*.jpg")) + list(batch_path.glob("*.JPG"))
            total_images += len(images)
            
            print(f"\nProcessing {batch_folder}: {len(images)} images")
            print(f"Output folder: {output_batch_path}")
            
            # Process each image in this batch
            batch_processed = 0
            for i, img_path in enumerate(images):
                try:
                    success = self.process_single_image(img_path, output_batch_path, batch_folder, num_augmentations)
                    if success:
                        batch_processed += 1
                        total_processed += 1
                    
                    # Progress update
                    if (i + 1) % 10 == 0 or (i + 1) == len(images):
                        print(f"  Progress: {i+1}/{len(images)} images processed")
                        
                except Exception as e:
                    print(f"  Error processing {img_path.name}: {e}")
                    continue
            
            print(f"  Completed {batch_folder}: {batch_processed}/{len(images)} images processed")
        
        # Save new annotation file
        self.save_augmented_annotations()
        
        # Final summary
        total_time = time.time() - start_time
        print(f"\n" + "=" * 60)
        print("AUGMENTATION COMPLETED!")
        print(f"Total images found: {total_images}")
        print(f"Total images processed: {total_processed}")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Average time per image: {total_time/total_processed:.1f} seconds")
        print(f"\nOutput structure:")
        print(f"  Main folder: {self.output_path}")
        print(f"  Each batch has its own subfolder")
        print(f"  Each image gets {num_augmentations + 1} versions (1 original + {num_augmentations} augmented)")
        print(f"  New annotation file: {self.output_path}/annotations_augmented.json")
        
        # Show output folder structure
        self.show_output_structure()
    
    def save_augmented_annotations(self):
        """Save the new annotation file with all augmented images"""
        annotation_file = self.output_path / "annotations_augmented.json"
        
        with open(annotation_file, 'w') as f:
            json.dump(self.augmented_annotations, f, indent=2)
        
        print(f"\nAnnotation file saved: {annotation_file}")
        print(f"Total images in annotations: {len(self.augmented_annotations['images'])}")
        print(f"Total annotations: {len(self.augmented_annotations['annotations'])}")
        print(f"Total categories: {len(self.augmented_annotations['categories'])}")
    
    def show_output_structure(self):
        """Display the output folder structure"""
        print(f"\nOutput folder structure:")
        print(f"  {self.output_path}/")
        
        for batch_folder in sorted(self.output_path.iterdir()):
            if batch_folder.is_dir():
                image_count = len(list(batch_folder.glob("*.jpg")))
                print(f"    {batch_folder.name}/ ({image_count} images)")
        
        print(f"    annotations_augmented.json")
    
    def get_statistics(self):
        """Get statistics about the processed datasets"""
        total_images = 0
        total_files = 0
        
        for batch_folder in self.output_path.iterdir():
            if batch_folder.is_dir():
                images = list(batch_folder.glob("*.jpg"))
                total_images += len(images)
                total_files += len(images)
        
        print(f"\nDataset Statistics:")
        print(f"  Total images: {total_images}")
        print(f"  Total files: {total_files}")
        print(f"  Output folder: {self.output_path}")
        print(f"  Annotation file: annotations_augmented.json")

def main():
    """Main function to run the augmentation"""
    print("Comprehensive Dataset Augmentation with Annotation Generation")
    print("=" * 60)
    
    # Initialize augmenter
    augmenter = DatasetAugmenter("data", "augmented_datasets")
    
    # Process all datasets
    augmenter.process_all_datasets(num_augmentations=2)
    
    # Show statistics
    augmenter.get_statistics()

if __name__ == "__main__":
    main()
