# Dataset Preparation Guide

This guide provides instructions on how to download and prepare the datasets required for running the continual learning experiments.

## Prerequisites

Before starting, ensure you have the following Python packages installed (beyond the base LLaVA-NeXT requirements):

```bash
pip install datasets requests tqdm pillow
```

For Hugging Face datasets, you may need to login:
```bash
pip install huggingface_hub
huggingface-cli login  # Follow prompts to enter your token
```

## General Setup

1.  **Create a data directory:** It's recommended to store all your datasets in a single location.
    ```bash
    export DATA_DIR=/path/to/your/data/directory
    mkdir -p $DATA_DIR
    ```

2.  **Image Folder:** The experiment scripts use an environment variable `IMAGE_FOLDER` to locate the images for all datasets. All dataset preparation scripts place images in subdirectories within a main image folder. Set this path in the experiment scripts. A good practice is to have a central image folder.
    ```bash
    export IMAGE_FOLDER=$DATA_DIR/images
    mkdir -p $IMAGE_FOLDER
    ```
    The experiment scripts in `scripts/all_experiments/final_experiments/` often include a placeholder for this path and for dataset YAMLs; update them to match your setup.

3.  **LLaVA-format JSONs:** The training scripts expect dataset information in a specific LLaVA JSON format. The scripts below will generate these JSON files. These JSONs are then referenced by `.yaml` files in `scripts/all_experiments/`.

4.  **Update YAML configurations:** After generating the JSON files, update the paths in the YAML files in `scripts/all_experiments/` to point to your actual data locations.

5.  **Security Note:** Avoid hardcoding API keys, tokens, or private paths.
    - Use environment variables or a local config file (gitignored) for sensitive info
    - Replace any example keys in utility scripts with your own via env vars (e.g., `FLICKR_API_KEY`, `FLICKR_API_SECRET`)
    - Update all placeholder paths to your actual data locations

---

## Dataset-specific Instructions

Below are instructions for preparing each dataset. The scripts will download the data and convert it to the required format.

### CUB-200

The CUB-200 dataset is a collection of bird images.

1.  **Run the conversion script:**
    The script downloads CUB-200-2011 if missing and converts to LLaVA format. Run it separately for each split:
    ```bash
    # Train split
    python utils/cub200_to_llava.py \
      --data_dir $DATA_DIR/datasets \
      --output_dir $DATA_DIR/llava_json \
      --image_dir $IMAGE_FOLDER/cub200 \
      --data_split train

    # Test split
    python utils/cub200_to_llava.py \
      --data_dir $DATA_DIR/datasets \
      --output_dir $DATA_DIR/llava_json \
      --image_dir $IMAGE_FOLDER/cub200 \
      --data_split test
    ```

2.  **Create `cub200.yaml`:**
    Create `scripts/all_experiments/cub200.yaml` with the following content (update the path to your generated JSON):
    ```yaml
    datasets:
    - json_path: /path/to/your/data/directory/llava_json/cub200_train.json
      sampling_strategy: "all"
    ```

### PixMo-Count

PixMo-Count is a dataset for object counting that requires downloading images from Flickr.

#### Step 1: Set up Flickr API (Highly Recommended)

Since the images are hosted on Flickr, you'll need Flickr API credentials to download them reliably:

1.  **Create a Flickr account** at [flickr.com](https://www.flickr.com) if you don't have one.
2.  **Apply for API access** at [Flickr's API page](https://www.flickr.com/services/api/misc.api_keys.html).
3.  **Get your API key and secret** from the Flickr API management page.

#### Step 2: Configure the download script

1.  **Update paths in the script:**
    Open `utils/download_pixmocount_and_convert_to_llava.py` and modify the paths at the bottom:
    ```python
    # In utils/download_pixmocount_and_convert_to_llava.py (bottom of file)
    if __name__ == "__main__":
        output_dir = "/path/to/your/data/directory/llava_json"
        image_dir = "/path/to/your/image/data/pixmo_count"
    ```

2.  **Provide Flickr API credentials (recommended):**
    Export them as environment variables (preferred), then modify the script to read from env if needed:
    ```bash
    export FLICKR_API_KEY="your_key"
    export FLICKR_API_SECRET="your_secret"
    ```
    The current script has inline placeholders; replace them with `os.environ.get("FLICKR_API_KEY")` and `os.environ.get("FLICKR_API_SECRET")`, or temporarily paste your keys locally (do not commit keys).

    Note: Without API keys, downloads may be rate limited.

#### Step 3: Run the download and conversion

```bash
python utils/download_pixmocount_and_convert_to_llava.py
```

This script will:
- Download the PixMo-Count dataset metadata from Hugging Face (`allenai/pixmo-count`)
- Download images from Flickr using the provided URLs
- Convert the data to LLaVA format and save as `pixmo_count_train.json`

#### Step 4: Create the YAML configuration

The YAML file should already be present at `scripts/all_experiments/pixmocount.yaml`, but verify it points to the correct location:
```yaml
datasets:
- json_path: /path/to/your/data/directory/llava_json/pixmo_count_train.json
  sampling_strategy: "all"
```

#### Troubleshooting

- **Rate limiting errors:** Ensure you have valid Flickr API credentials.
- **Download failures:** Some images may no longer be available on Flickr. The script includes retry logic and will skip unavailable images.
- **Large download:** The dataset contains thousands of images, so the download may take several hours.

### PathVQA

The PathVQA dataset contains pathology images with questions and answers.

1.  **Run the conversion script:**
    ```bash
    # e.g., for the train split
    python utils/pathvqa_to_llava.py --output_dir $DATA_DIR/llava_json --image_dir $IMAGE_FOLDER/pathvqa --data_split train
    ```
    Repeat with `--data_split validation` and `--data_split test` as needed. Images are saved and JSONs are created per split.

2.  **Create `pathvqa.yaml`:**
    ```yaml
    datasets:
    - json_path: /path/to/your/data/directory/llava_json/pathvqa_train.json
      sampling_strategy: "all"
    ```

### TextVQA

The TextVQA dataset requires reading text in images to answer questions.

1.  **Run the conversion script:**
    ```bash
    # e.g., for the train split
    python utils/textvqa_to_llava.py --output_dir $DATA_DIR/llava_json --image_dir $IMAGE_FOLDER/textvqa --data_split train
    ```
    This downloads the dataset from Hugging Face and creates a LLaVA JSON for the selected split.

2.  **Create `textvqa.yaml`:**
    ```yaml
    datasets:
    - json_path: /path/to/your/data/directory/llava_json/textvqa_train.json
      sampling_strategy: "all"
    ```

### TimeClock and Related Clock Datasets

For evaluation, no local conversion is required. The tasks load curated datasets directly from Hugging Face:
- `timeclock`: `AvaXiao/clockreading-time`
- `cococlock`: `Jessemel/clockreading-coco`
- `openimgclock`: `Jessemel/clockreading-openimg`

Ensure `huggingface-cli login` is configured if any dataset requires gated access.

Optional (advanced): If you want to build custom clock-reading datasets, `utils/clockreading_to_llava.py` shows an example conversion pipeline. Note that it currently contains hardcoded paths; edit `base_dir` and `output_dir` in the script before running.

<!-- Rehearsal-based approaches are deprecated and have been removed. -->
