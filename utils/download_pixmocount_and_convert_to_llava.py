import json
import os
from datasets import load_dataset
import hashlib
import requests
from tqdm import tqdm
import time
import re

# Replace these with your Flickr API keys
FLICKR_API_KEY = "2314c4da61d8dcaad2d3051c8e8b91ae"
FLICKR_API_SECRET = "f80c8e5bd42d35a3"

def extract_photo_id(url):
    """Extract photo ID from Flickr URL."""
    pattern = r'/(\d+)_[a-zA-Z0-9]+_[a-zA-Z]\.jpg$'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    print(f"Could not extract photo ID from URL: {url}")
    return None

def get_photo_url(photo_id):
    """Get original photo URL using Flickr API."""
    api_url = f"https://api.flickr.com/services/rest/?method=flickr.photos.getSizes&api_key={FLICKR_API_KEY}&photo_id={photo_id}&format=json&nojsoncallback=1"
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            if data['stat'] == 'ok':
                # Get the largest available size
                sizes = data['sizes']['size']
                return sizes[-1]['source']
            else:
                print(f"Flickr API error for photo {photo_id}: {data.get('message', 'Unknown error')}")
        else:
            print(f"HTTP error {response.status_code} when accessing Flickr API for photo {photo_id}")
    except Exception as e:
        print(f"Error accessing Flickr API for photo {photo_id}: {str(e)}")
    return None

def download_with_retry(url, max_retries=5):
    """Download with exponential backoff for 429 errors."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 429:
                wait_time = (2 ** attempt) * 5  # 5, 10, 20, 40, 80 seconds
                print(f"Rate limited (429). Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
                continue
            return response
        except Exception as e:
            print(f"Error during retry {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(5)
            continue
    return None

def download_image(url, image_sha256, image_dir):
    """Download image with rate limiting."""
    try:
        # Rate limiting - 1.1 seconds per request
        time.sleep(1.1)
        
        image_path = f"{image_sha256}.jpg"
        full_path = os.path.join(image_dir, image_path)
        
        # Try to get original URL through API if it's a Flickr URL
        photo_id = extract_photo_id(url)
        if photo_id:
            print(f"Found Flickr photo ID: {photo_id}")
            api_url = get_photo_url(photo_id)
            if api_url:
                print(f"Using Flickr API URL: {api_url}")
                url = api_url
            else:
                print(f"Failed to get API URL for photo {photo_id}, using original URL")
        
        # Download image with retry
        print(f"Downloading from URL: {url}")
        response = download_with_retry(url)
        if response is None:
            print("Failed after all retries")
            return None
            
        if response.status_code != 200:
            print(f"HTTP error {response.status_code} when downloading image")
            print(f"Response headers: {dict(response.headers)}")
            return None
            
        # Save image
        content_length = len(response.content)
        print(f"Downloaded {content_length} bytes")
        
        with open(full_path, "wb") as f:
            f.write(response.content)
            
        if os.path.exists(full_path):
            file_size = os.path.getsize(full_path)
            if file_size > 0:
                print(f"Successfully saved image ({file_size} bytes)")
                return image_path
            else:
                print("Saved file is empty")
        else:
            print("Failed to save file")
        return None
    except requests.exceptions.Timeout:
        print(f"Timeout error downloading {url}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"Connection error downloading {url}")
        return None
    except Exception as e:
        print(f"Unexpected error downloading {url}: {str(e)}")
        return None

def convert_to_llava_format(output_dir, image_dir, split="train"):
    # Load dataset
    print(f"\nProcessing {split} split...")
    try:
        dataset = load_dataset("allenai/pixmo-count", split=split)
    except Exception as e:
        print(f"Failed to load dataset: {str(e)}")
        return
    
    print(f"Dataset size: {len(dataset)}")
    print(f"First item example: {dataset[0]}")
    
    # Check for existing progress
    temp_file = os.path.join(output_dir, f"pixmo_count_{split}_temp.json")
    if os.path.exists(temp_file):
        print(f"Found existing progress file: {temp_file}")
        with open(temp_file, "r") as f:
            llava_data = json.load(f)
        processed_ids = {item["id"] for item in llava_data}
        print(f"Resuming from {len(processed_ids)} previously processed items")
    else:
        llava_data = []
        processed_ids = set()
    
    success_count = len(processed_ids)
    fail_count = 0
    
    # Process each item
    for idx in tqdm(range(len(dataset)), desc=f"Processing {split} split"):
        # Skip if already processed
        current_id = dataset[idx]["image_sha256"]
        if current_id in processed_ids:
            print(f"\nSkipping already processed item {idx} with id {current_id}")
            continue
        
        print(f"\nProcessing item {idx}")
        # Download image
        image_path = download_image(
            dataset[idx]['image_url'],
            dataset[idx]['image_sha256'],
            image_dir
        )
        
        if image_path is None:
            print(f"Failed to download image {idx}")
            fail_count += 1
            continue
            
        # Create conversation
        conversation = {
            "id": current_id,
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": f"How many {dataset[idx]['label']} are in this image?\n<image>"
                },
                {
                    "from": "gpt",
                    "value": f"There {'is' if dataset[idx]['count'] == 1 else 'are'} {dataset[idx]['count']} {dataset[idx]['label']} in this image."
                }
            ]
        }
        
        llava_data.append(conversation)
        processed_ids.add(current_id)
        success_count += 1
        print(f"Successfully processed item {idx}")
        
        # Save progress every 100 images
        if success_count % 100 == 0:
            with open(temp_file, "w") as f:
                json.dump(llava_data, f, indent=2)
            print(f"Saved progress after {success_count} successful downloads")
    
    # Print statistics
    print(f"\nConversion statistics for {split} split:")
    print(f"Total items: {len(dataset)}")
    print(f"Successfully downloaded: {success_count}")
    print(f"Failed downloads: {fail_count}")
    print(f"Success rate: {success_count/len(dataset)*100:.2f}%")
    
    # Save to JSON file
    print(f"\nSaving {split} data...")
    output_file = os.path.join(output_dir, f"pixmo_count_{split}.json")
    with open(output_file, "w") as f:
        json.dump(llava_data, f, indent=2)
    print(f"Successfully saved {len(llava_data)} conversations to {output_file}")
    
    # Clean up temp file only if all items were processed successfully
    if len(processed_ids) == len(dataset):
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"Removed temporary file as all items were processed")

if __name__ == "__main__":
    output_dir = "/work/nvme/bcgq/zhenzhu/data/llava_data/pixmo_count"
    image_dir = "/work/nvme/bcgq/zhenzhu/data/llava_data/pixmo_count/images"
    
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting PixMo-Count to LLaVA conversion...")
    
    # Convert train split
    convert_to_llava_format(output_dir, image_dir, "train")
    
    # Convert validation split 
    convert_to_llava_format(output_dir, image_dir, "validation")
    
    # Convert test split
    convert_to_llava_format(output_dir, image_dir, "test")
    
    print("\nConversion completed!")
