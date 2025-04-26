import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from collections import Counter
import cv2
import os
import argparse

def analyze_coco_dataset(annotation_file, img_dir, output_dir):
    """
    Analyze a COCO dataset to get class distribution and visualize examples
    
    Args:
        annotation_file (str): Path to the COCO annotation JSON file
        img_dir (str): Directory containing the images
    """
    # Load COCO annotations
    coco = COCO(annotation_file)

    img_ids = coco.getImgIds()
    instance_indicators = 0
    semantic_indicators = 0

    for img_id in img_ids[:5]:  # Check first 5 images
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Check instances of same category
        cat_ids = [ann['category_id'] for ann in anns]
        unique_cats = set(cat_ids)
        
        # Instance segmentation often has multiple objects per image
        if len(anns) > len(unique_cats):
            instance_indicators += 1
        
        # Check image coverage (semantic tends to cover whole image)
        img_info = coco.loadImgs(img_id)[0]
        img_area = img_info['height'] * img_info['width']
        total_ann_area = sum(ann.get('area', 0) for ann in anns)
        
        # Semantic segmentation typically covers most/all of the image
        if total_ann_area >= 0.9 * img_area:
            semantic_indicators += 1

    # Make determination based on indicators
    if instance_indicators > semantic_indicators:
        print("Likely instance segmentation")
    else:
        print("Likely semantic segmentation")
    
    # Print dataset summary
    print(f"Number of images: {len(coco.getImgIds())}")
    print(f"Number of categories: {len(coco.getCatIds())}")
    print(f"Number of annotations: {len(coco.getAnnIds())}")
    
    # Get category information
    categories = coco.loadCats(coco.getCatIds())
    cat_names = {cat['id']: cat['name'] for cat in categories}
    
    # ----- CLASS DISTRIBUTION ANALYSIS -----
    
    # Method 1: Count instances per category (for instance segmentation)
    category_counts = Counter()
    for ann in coco.anns.values():
        category_counts[cat_names[ann['category_id']]] += 1
    
    # Plot category distribution
    plt.figure(figsize=(12, 6))
    categories_sorted = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in categories_sorted]
    counts = [x[1] for x in categories_sorted]
    
    plt.bar(range(len(names)), counts)
    plt.xticks(range(len(names)), names, rotation=90)
    plt.title('Category Distribution')
    plt.xlabel('Category')
    plt.ylabel('Number of Instances')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_distribution.png'))
    plt.close()
    
    print("\nCategory Distribution:")
    for name, count in categories_sorted:
        print(f"{name}: {count} instances")
    
    # Method 2: Calculate pixel distribution (for semantic segmentation)
    # This approach works better for semantic segmentation
    pixel_counts = Counter()
    sample_count = min(50, len(coco.getImgIds()))  # Analyze a subset for efficiency
    
    for i, img_id in enumerate(coco.getImgIds()[:sample_count]):
        # Get all annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # Load image info to get dimensions
        img_info = coco.loadImgs(img_id)[0]
        h, w = img_info['height'], img_info['width']
        
        # Create a blank semantic mask
        semantic_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Fill in the semantic mask
        for ann in anns:
            mask = coco.annToMask(ann)
            semantic_mask[mask == 1] = ann['category_id']
        
        # Count pixels per class
        unique, counts = np.unique(semantic_mask, return_counts=True)
        for cat_id, count in zip(unique, counts):
            if cat_id > 0:  # Skip background (usually 0)
                pixel_counts[cat_names.get(cat_id, f"Unknown-{cat_id}")] += count
            else:
                pixel_counts["Background"] += count
                
        # Visualize an example image and mask
        if i == 0:
            visualize_semantic_segmentation(coco, img_id, img_dir, semantic_mask, cat_names, output_dir)
    
    return coco

def visualize_semantic_segmentation(coco, img_id, img_dir, semantic_mask, cat_names, output_dir='output'):
    """
    Visualize an image and its semantic segmentation mask
    
    Args:
        coco: COCO API object
        img_id: Image ID to visualize
        img_dir: Directory containing images
        semantic_mask: Generated semantic mask
        cat_names: Dictionary mapping category IDs to names
    """
    # Load image info
    img_info = coco.loadImgs(img_id)[0]
    img_path = os.path.join(img_dir, img_info['file_name'])
    
    # Load the image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create a color mask for visualization
    color_mask = np.zeros((semantic_mask.shape[0], semantic_mask.shape[1], 3), dtype=np.uint8)
    
    # Generate random colors for each class
    unique_classes = np.unique(semantic_mask)
    colors = {}
    for cls in unique_classes:
        if cls == 0:  # Background
            colors[cls] = [0, 0, 0]  # Black
        else:
            # Generate random color
            colors[cls] = [np.random.randint(0, 255) for _ in range(3)]
    
    # Apply colors to mask
    for cls in unique_classes:
        color_mask[semantic_mask == cls] = colors[cls]
    
    # Create legend for the classes
    legend_patches = []
    for cls in unique_classes:
        if cls > 0:  # Skip background
            class_name = cat_names.get(cls, f"Unknown-{cls}")
            color = [c/255 for c in colors[cls]]  # Convert to 0-1 range for matplotlib
            legend_patches.append(plt.Rectangle((0,0), 1, 1, fc=color, label=class_name))
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot semantic mask
    axes[1].imshow(color_mask)
    axes[1].set_title('Semantic Segmentation')
    axes[1].axis('off')
    
    # Plot overlay
    overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    # Add legend
    if legend_patches:
        fig.legend(handles=legend_patches, loc='lower center', ncol=min(5, len(legend_patches)))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'semantic_segmentation_{img_id}.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze COCO dataset")
    parser.add_argument('--annotation_file', type=str, required=True, help='Path to COCO annotation file')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save output files')
    args = parser.parse_args()
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    # Set the output directory for saving plots
    annotation_file = args.annotation_file  # Path to COCO annotation file
    img_dir = args.img_dir  # Directory containing images
    
    coco = analyze_coco_dataset(annotation_file, img_dir, args.output_dir)