import json
import os
from datasets import load_dataset
import base64
from PIL import Image
import io
from tqdm import tqdm

def save_image(image_data, image_dir, idx):
    """Save image data to a file and return the filename."""
    try:
        # Create filename based on index
        image_filename = f"countbench_{idx:05d}.jpg"
        image_path = os.path.join(image_dir, image_filename)
        
        # Save the PIL Image
        if isinstance(image_data, Image.Image):
            image_data.save(image_path, "JPEG")
        else:
            # Convert to PIL Image if it's bytes
            Image.open(io.BytesIO(image_data)).save(image_path, "JPEG")
        
        # Verify the image was saved correctly
        if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
            print(f"Successfully saved image to {image_path}")
            return image_filename
        else:
            print(f"Failed to save image {idx}: File is empty or doesn't exist")
            return None
            
    except Exception as e:
        print(f"Error saving image {idx}: {str(e)}")
        return None

def convert_to_llava_format(output_dir, image_dir):
    """Convert CountBenchQA dataset to LLaVA format."""
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    print("\nLoading CountBenchQA dataset...")
    try:
        dataset = load_dataset("vikhyatk/CountBenchQA", split="test")
        print(f"Successfully loaded dataset with {len(dataset)} items")
    except Exception as e:
        print(f"Failed to load dataset: {str(e)}")
        return
    
    # Initialize data structures
    llava_data = []
    success_count = 0
    fail_count = 0
    
    # Process each item
    for idx in tqdm(range(len(dataset)), desc="Processing dataset"):
        print(f"\nProcessing item {idx}")
        try:
            # Save image
            image_path = save_image(dataset[idx]['image'], image_dir, idx)
            if image_path is None:
                print(f"Failed to save image for item {idx}")
                fail_count += 1
                continue
            
            # Create conversation
            conversation = {
                "id": f"countbench_{idx}",
                "image": image_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"{dataset[idx]['question']}\n<image>"
                    },
                    {
                        "from": "gpt",
                        "value": f"{dataset[idx]['number']}"
                    }
                ]
            }
            
            llava_data.append(conversation)
            success_count += 1
            print(f"Successfully processed item {idx}")
            
            # Save progress every 100 images
            if success_count % 100 == 0:
                temp_file = os.path.join(output_dir, "countbench_temp.json")
                with open(temp_file, "w") as f:
                    json.dump(llava_data, f, indent=2)
                print(f"Saved progress after {success_count} successful conversions")
                
        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            fail_count += 1
            continue
    
    # Print statistics
    print("\nConversion statistics:")
    print(f"Total items: {len(dataset)}")
    print(f"Successfully converted: {success_count}")
    print(f"Failed conversions: {fail_count}")
    print(f"Success rate: {success_count/len(dataset)*100:.2f}%")
    
    # Save final output
    print("\nSaving final output...")
    output_file = os.path.join(output_dir, "countbench_test.json")
    with open(output_file, "w") as f:
        json.dump(llava_data, f, indent=2)
    print(f"Saved final output to {output_file}")

if __name__ == "__main__":
    # Define output directories
    output_dir = "/work/nvme/bcgq/zhenzhu/data/llava_data/countbench"
    image_dir = os.path.join(output_dir, "images")
    
    # Convert dataset
    convert_to_llava_format(output_dir, image_dir)
