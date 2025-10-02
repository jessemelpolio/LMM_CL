import json
import csv
import os
from pathlib import Path

def convert_dataset_to_llava(csv_path, output_path, dataset_name):
    llava_data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {
                "id": str(row['img_id']),
                "image": f"{row['file_name']}",
                "conversations": [
                    {
                        "from": "human",
                        "value": "What time does the clock show in this image?\n<image>"
                    },
                    {
                        "from": "gpt",
                        "value": f"{int(row['hour']):02d}:{int(row['minute']):02d}"
                    }
                ]
            }
            llava_data.append(entry)
    
    with open(output_path, 'w') as f:
        json.dump(llava_data, f, indent=2)

def main():
    base_dir = Path("/work/nvme/bcgq/zhenzhu/experiments/LLaVA-NeXT/utils/itsabouttime/data")
    output_dir = Path("/work/nvme/bcgq/zhenzhu/data/llava_data/clockreading")
    
    datasets = [
        ("coco_final.csv", "coco_test.json", "coco"),
        ("movies_final.csv", "clockmovies_test.json", "clockmovies"),
        ("openimg_final.csv", "openimg_test.json", "openimg")
    ]
    
    for csv_file, output_file, dataset_name in datasets:
        csv_path = base_dir / csv_file
        output_path = output_dir / dataset_name / output_file
        convert_dataset_to_llava(csv_path, output_path, dataset_name)
        print(f"Converted {dataset_name} dataset to {output_path}")

if __name__ == "__main__":
    main()