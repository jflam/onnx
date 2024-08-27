import os
import sys
import csv
import time
from argparse import ArgumentParser
from huggingface_hub import HfApi
from tenacity import retry, stop_after_attempt, wait_exponential
from huggingface_hub.utils._errors import HfHubHTTPError

# Initialize the Hugging Face API
api = HfApi(token=os.environ.get("HF_TOKEN"))

DEFAULT_MODELS = [
    "microsoft/phi-3.5-mini",
    "microsoft/Phi-3.5-vision-instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-mini-128k-instruct",
    "microsoft/phi-2",
    "google/gemma-2-9b-it",
    "google/gemma-2-2b-it",
    "google/gemma-2b",
    "google/gemma-1.1-7b-it",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Codestral-22B-v0.1",
    "Qwen/Qwen2-7B-Instruct",
]

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
def list_models_with_retry(search_name):
    try:
        return api.list_models(search=search_name, full=True)
    except HfHubHTTPError as e:
        print(f"HfHubHTTPError occurred: {e}")
        raise  # Re-raise the exception to trigger a retry
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        raise  # Re-raise the exception to trigger a retry

def find_model_variants(base_model_name):
    variants = {
        "original": [],
        "onnx": [],
        "gguf": []
    }
    
    # Remove organization name from the search query
    search_name = base_model_name.split('/')[-1]
    
    # Search for the base model and its variants
    try:
        models = list_models_with_retry(search_name)
    except Exception as e:
        print(f"Error searching for {base_model_name}: {str(e)}")
        return None

    try:
        for model in models:
            repo_id = model.modelId.lower()
            model_name = repo_id.split('/')[-1]
            
            if search_name.lower().replace('-', '') in model_name.replace('-', ''):
                if "onnx" in model_name:
                    variants["onnx"].append(model)
                elif "gguf" in model_name:
                    variants["gguf"].append(model)
                else:
                    variants["original"].append(model)
    except Exception as e:
        print(f"Error processing models for {base_model_name}: {str(e)}")
        return None

    return variants

def calculate_downloads(variants):
    downloads = {
        "original": 0,
        "onnx": 0,
        "gguf": 0
    }
    
    for variant_type, models in variants.items():
        downloads[variant_type] = sum(model.downloads for model in models)
    
    return downloads

def print_results(base_model_name, variants, downloads, csv_filename=None):
    print(f"\nResults for {base_model_name}:")
    print(f"Original models: {len(variants['original'])} (Total downloads: {downloads['original']})")
    print(f"ONNX models: {len(variants['onnx'])} (Total downloads: {downloads['onnx']})")
    print(f"GGUF models: {len(variants['gguf'])} (Total downloads: {downloads['gguf']})")

    if csv_filename:
        with open(csv_filename, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                base_model_name,
                len(variants['original']), downloads['original'],
                len(variants['onnx']), downloads['onnx'],
                len(variants['gguf']), downloads['gguf']
            ])

def main():
    parser = ArgumentParser(description="Analyze Hugging Face model variants and their download counts.")
    parser.add_argument("--model", help="Specific model to analyze (optional)")
    parser.add_argument("--csv", action="store_true", help="Output results in CSV format")
    args = parser.parse_args()

    models_to_analyze = [args.model] if args.model else DEFAULT_MODELS

    csv_filename = None

    if args.csv:
        csv_filename = 'model_analysis_results.csv'
        with open(csv_filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Base Model', 'Original Count', 'Original Downloads', 'ONNX Count', 'ONNX Downloads', 'GGUF Count', 'GGUF Downloads'])

    for base_model_name in models_to_analyze:
        print(f"\nAnalyzing {base_model_name}...")
        variants = find_model_variants(base_model_name)
        
        if variants is None:
            print(f"Skipping {base_model_name} due to error.")
            continue

        downloads = calculate_downloads(variants)
        print_results(base_model_name, variants, downloads, csv_filename)

    if csv_filename:
        print(f"\nResults have been saved to {csv_filename}")

if __name__ == "__main__":
    main()