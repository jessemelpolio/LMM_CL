import json
import os
import argparse
import random
import shutil
import re
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import requests
import tarfile
from io import BytesIO
import numpy as np

# Question templates to randomize during creation
QUESTION_TEMPLATES = [
    "What species is the bird in this photo?",
    "Can you identify the bird species in this image?",
    "Which bird species is shown in this picture?", 
    "What type of bird is depicted in this photograph?",
    "Identify the species of the bird in this image.",
    "What is the bird species presented in this photo?",
    "Which of the following species does this bird belong to?",
    "What would you classify this bird as?",
    "From the options below, what species is this bird?",
    "Looking at this bird, what species would you say it is?"
]

def download_cub_dataset(data_dir):
    """Download CUB-200-2011 dataset if not already downloaded."""
    url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
    extract_dir = os.path.join(data_dir, "CUB_200_2011")
    temp_file = os.path.join(data_dir, "CUB_200_2011.tgz")
    
    # Skip if already downloaded and extracted
    if os.path.exists(extract_dir):
        print(f"CUB-200-2011 dataset already exists at {extract_dir}")
        return extract_dir
    
    # Create directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Download the dataset with progress bar
    print(f"Downloading CUB-200-2011 dataset...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get the file size
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 8192  # 8 Kibibytes
    
    # Create progress bar
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    # Download the file in chunks, showing progress
    with open(temp_file, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()
    
    # Check if download completed successfully
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR: Download incomplete or corrupted")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return None
    
    # Extract the tar file with progress
    print(f"Extracting dataset to {data_dir}...")
    with tarfile.open(temp_file, mode="r:gz") as tar:
        members = tar.getmembers()
        extract_progress = tqdm(members, desc="Extracting files")
        for member in extract_progress:
            tar.extract(member, path=data_dir)
    
    # Clean up the tar file
    os.remove(temp_file)
    
    print(f"Downloaded and extracted CUB-200-2011 dataset to {extract_dir}")
    return extract_dir

def load_class_names(dataset_dir):
    """Load class names from the dataset."""
    classes_file = os.path.join(dataset_dir, "classes.txt")
    class_names = {}
    
    with open(classes_file, "r") as f:
        for line in f:
            class_id, class_name = line.strip().split(" ", 1)
            
            # Replace underscores with spaces for readability
            class_name = class_name.replace("_", " ")
            
            # Remove the numerical prefix (e.g., "067.")
            # The format in the file is typically "###.BirdName"
            if "." in class_name:
                prefix, bird_name = class_name.split(".", 1)
                # Check if the part before the period is a number
                if prefix.isdigit():
                    class_name = bird_name.strip()
            
            class_names[int(class_id)] = class_name
            
    return class_names

def load_image_class_labels(dataset_dir):
    """Load image to class mappings."""
    image_class_file = os.path.join(dataset_dir, "image_class_labels.txt")
    image_classes = {}
    
    with open(image_class_file, "r") as f:
        for line in f:
            image_id, class_id = line.strip().split()
            image_classes[int(image_id)] = int(class_id)
            
    return image_classes

def load_image_paths(dataset_dir):
    """Load image paths."""
    images_file = os.path.join(dataset_dir, "images.txt")
    image_paths = {}
    
    with open(images_file, "r") as f:
        for line in f:
            image_id, image_path = line.strip().split()
            image_paths[int(image_id)] = image_path
            
    return image_paths

def load_train_test_split(dataset_dir):
    """Load the official train/test split information."""
    split_file = os.path.join(dataset_dir, "train_test_split.txt")
    splits = {}
    
    with open(split_file, "r") as f:
        for line in f:
            image_id, is_training = line.strip().split()
            # 1 indicates training set, 0 indicates test set
            splits[int(image_id)] = int(is_training)
            
    return splits

def create_multiple_choice_question(class_names, correct_class_id):
    """Create a multiple choice question with 26 options (A-Z)."""
    # Get all possible class IDs (1-200)
    all_class_ids = list(class_names.keys())
    
    # Choose the correct class and the letter it will be assigned to (A-Z)
    correct_letter_idx = random.randint(0, 25)  # 0-25 corresponding to A-Z
    correct_letter = chr(65 + correct_letter_idx)  # Convert to A-Z
    
    # Select 25 incorrect classes randomly
    incorrect_class_ids = [cid for cid in all_class_ids if cid != correct_class_id]
    incorrect_class_ids = random.sample(incorrect_class_ids, 25)
    
    # Arrange all 26 options with the correct one in its random position
    options = []
    for i in range(26):
        letter = chr(65 + i)  # A, B, C, ..., Z
        if i == correct_letter_idx:
            # Add the correct class
            options.append(f"{letter}. {class_names[correct_class_id]}")
        else:
            # Add an incorrect class
            incorrect_idx = i if i < correct_letter_idx else i - 1
            options.append(f"{letter}. {class_names[incorrect_class_ids[incorrect_idx]]}")
    
    # Build the full question text
    question_template = random.choice(QUESTION_TEMPLATES)
    question = f"{question_template}\nAnswer with the option's letter from the given choices directly.\n"
    question += "\n".join(options)
    
    return question, correct_letter

def save_image(src_path, dest_dir, image_id):
    """Copy image from source to destination directory."""
    try:
        # Create filename based on index
        image_filename = f"cub200_{image_id:05d}.jpg"
        dest_path = os.path.join(dest_dir, image_filename)
        
        # Skip if image already exists
        if os.path.exists(dest_path):
            return image_filename
        
        # Copy the image file
        shutil.copy2(src_path, dest_path)
        
        return image_filename
    except Exception as e:
        print(f"Error copying image {image_id}: {e}")
        return None

def convert_to_llava_format(dataset_dir, output_dir, image_dir, data_split="train"):
    """Convert CUB-200-2011 dataset to LLaVA multiple-choice format using the official split."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    # Load class names, image labels, and paths
    class_names = load_class_names(dataset_dir)
    image_classes = load_image_class_labels(dataset_dir)
    image_paths = load_image_paths(dataset_dir)
    train_test_split = load_train_test_split(dataset_dir)
    
    # Filter image IDs based on the official split and the requested split
    if data_split == "train":
        # Training set has value 1 in the train_test_split file
        image_ids = [img_id for img_id, is_train in train_test_split.items() if is_train == 1]
    elif data_split == "test":
        # Test set has value 0 in the train_test_split file
        image_ids = [img_id for img_id, is_train in train_test_split.items() if is_train == 0]
    else:
        # Use all images if not train or test
        image_ids = list(image_paths.keys())
    
    print(f"Processing {data_split} split with {len(image_ids)} images...")
    
    # List to store all conversation data
    all_data = []
    
    # Process images and create conversation data
    for idx, image_id in enumerate(tqdm(image_ids)):
        # Get class and path for this image
        class_id = image_classes[image_id]
        image_path = os.path.join(dataset_dir, "images", image_paths[image_id])
        
        # Save/copy the image
        image_filename = save_image(image_path, image_dir, image_id)
        
        if not image_filename:
            print(f"Skipping image {image_id} due to save failure")
            continue
        
        # Create multiple-choice question and get correct answer
        question, correct_answer = create_multiple_choice_question(class_names, class_id)
        
        # Create conversation format
        conversation_data = {
            "id": f"cub200_{image_id:05d}",
            "image": f"cub200/images/{image_filename}",
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{question}"
                },
                {
                    "from": "gpt",
                    "value": correct_answer
                }
            ]
        }
        
        all_data.append(conversation_data)
    
    # Save to JSON
    output_file = os.path.join(output_dir, f"cub200_{data_split}.json")
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"Conversion complete. {len(all_data)} entries saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert CUB-200-2011 dataset to LLaVA format with multiple-choice questions")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory for JSON file")
    parser.add_argument("--image_dir", type=str, default="./data/cub200", help="Directory to save images")
    parser.add_argument("--data_dir", type=str, default="./data/datasets", help="Directory to download/store the original dataset")
    parser.add_argument("--data_split", type=str, default="train", choices=["train", "test", "all"], 
                        help="Dataset split to convert (using the official CUB split)")
    
    args = parser.parse_args()
    
    # Download/load the CUB-200-2011 dataset
    dataset_dir = download_cub_dataset(args.data_dir)
    
    # Convert to LLaVA format
    convert_to_llava_format(
        dataset_dir, 
        args.output_dir, 
        args.image_dir, 
        args.data_split
    )

if __name__ == "__main__":
    main() 