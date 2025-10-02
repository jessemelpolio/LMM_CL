import json
import os
import argparse
from collections import Counter
from datasets import load_dataset
from PIL import Image
from io import BytesIO
from tqdm import tqdm

def save_image(image_data, image_dir, image_id):
    """Save image data to a file and return the filename."""
    try:
        # Create filename based on index
        image_filename = f"textvqa_{image_id}.jpg"
        image_path = os.path.join(image_dir, image_filename)
        
        # Skip if image already exists
        if os.path.exists(image_path):
            return image_filename
        
        # Save the PIL Image
        if isinstance(image_data, Image.Image):
            image_data.save(image_path, "JPEG")
        else:
            # This shouldn't happen with the current dataset, but just in case
            Image.open(BytesIO(image_data)).save(image_path, "JPEG")
        
        return image_filename
    except Exception as e:
        print(f"Error saving image {image_id}: {e}")
        return None

def get_most_common_answer(answers):
    """Get the most common answer from the list of answers."""
    if not answers:
        return "No answer available"
    
    # Count occurrences of each answer
    answer_counts = Counter(answers)
    
    # Return the most common answer
    return answer_counts.most_common(1)[0][0]

def convert_to_llava_format(dataset, output_dir, image_dir, data_split="train"):
    """Convert TextVQA dataset to LLaVA format."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    # List to store all conversation data
    all_data = []
    
    # Process dataset and create conversation data
    print(f"Processing {data_split} split...")
    for idx, item in enumerate(tqdm(dataset)):
        image_id = item['image_id']
        question = item['question']
        answers = item['answers']
        image_data = item['image']
        
        # Save the image
        image_filename = save_image(image_data, image_dir, image_id)
        
        if not image_filename:
            print(f"Skipping item {idx} due to image save failure")
            continue
        
        # Get the most common answer
        answer = get_most_common_answer(answers)
        
        # Create conversation format
        conversation_data = {
            "id": f"textvqa_{image_id}",
            "image": f"textvqa/images/{image_filename}",
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{question}"
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        }
        
        all_data.append(conversation_data)
    
    # Save to JSON
    output_file = os.path.join(output_dir, f"textvqa_{data_split}.json")
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"Conversion complete. {len(all_data)} entries saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert TextVQA dataset to LLaVA format")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory for JSON file")
    parser.add_argument("--image_dir", type=str, default="./data/textvqa", help="Directory to save images")
    parser.add_argument("--data_split", type=str, default="train", choices=["train", "validation", "test"], 
                        help="Dataset split to convert")
    
    args = parser.parse_args()
    
    # Load TextVQA dataset
    print(f"Loading TextVQA dataset {args.data_split} split...")
    dataset = load_dataset("lmms-lab/textvqa", split=args.data_split)
    
    # Convert to LLaVA format
    convert_to_llava_format(
        dataset, 
        args.output_dir, 
        args.image_dir, 
        args.data_split
    )

if __name__ == "__main__":
    main() 