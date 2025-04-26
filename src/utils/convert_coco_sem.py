from ast import parse
import json
import os
from pycocotools.coco import COCO
from tqdm import tqdm
import argparse

def simplify_coco_categories(input_annotation_file, output_annotation_file, new_label_name="Main"):
    """
    Simplify a COCO dataset by changing all category IDs to 1 and setting
    a single new label name for all annotations.
    
    Args:
        input_annotation_file (str): Path to the input COCO annotation JSON file
        output_annotation_file (str): Path to save the modified annotation file
        new_label_name (str): New name for the single category
    """
    # Load COCO annotations
    print(f"Loading COCO annotations from {input_annotation_file}...")
    with open(input_annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create a new categories list with just one category
    coco_data['categories'] = [{
        "id": 1,
        "name": new_label_name,
        "supercategory": "object"
    }]
    
    # Update all annotations to have category_id = 1
    print("Updating category IDs in annotations...")
    for ann in tqdm(coco_data['annotations']):
        ann['category_id'] = 1
    
    # Save the modified annotation file
    print(f"Saving modified annotations to {output_annotation_file}...")
    with open(output_annotation_file, 'w') as f:
        json.dump(coco_data, f)
    
    print(f"Done! All categories changed to '{new_label_name}' with ID 1")
    print(f"Modified {len(coco_data['annotations'])} annotations")
    
    return coco_data

if __name__ == "__main__":
    # Set your paths here
    parser = argparse.ArgumentParser(description="Simplify COCO dataset categories")
    parser.add_argument("--input_annotation_file", type=str, required=True, help="Path to input COCO annotation file")
    parser.add_argument("--output_annotation_file", type=str, required=True, help="Path to output COCO annotation file")
    parser.add_argument("--new_label_name", type=str, default="Main", help="New name for all categories")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.input_annotation_file), exist_ok=True)
    input_annotation_file = args.input_annotation_file  # Input annotation file
    output_annotation_file = args.output_annotation_file  # Output annotation file
    new_label_name = args.new_label_name  # New label name for all categories
    
    simplify_coco_categories(
        input_annotation_file=input_annotation_file,
        output_annotation_file=output_annotation_file,
        new_label_name=new_label_name
    )