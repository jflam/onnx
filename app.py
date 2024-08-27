import os
from huggingface_hub import HfApi

# Initialize the Hugging Face API
api = HfApi(token=os.environ.get("HF_TOKEN"))

def find_model_variants(base_model_name):
    variants = {
        "original": [],
        "onnx": [],
        "gguf": [],
        "safetensors": []
    }
    
    # Remove organization name from the search query
    search_name = base_model_name.split('/')[-1]
    
    # Search for the base model and its variants
    models = api.list_models(search=search_name, full=True)
    
    print(f"\nRaw models returned from the API (searching for '{search_name}'):")
    for model in models:
        print(f"Model ID: {model.modelId}")
        print(f"URL: https://huggingface.co/{model.modelId}")
        print("-" * 50)
    
        repo_id = model.modelId.lower()
        model_name = repo_id.split('/')[-1]
        
        if search_name.lower().replace('-', '') in model_name.replace('-', ''):
            if "onnx" in model_name:
                variants["onnx"].append(model.modelId)
            elif "gguf" in model_name:
                variants["gguf"].append(model.modelId)
            else:
                variants["original"].append(model.modelId)
            
            # Check for safetensors in the file list
            if any(file.rfilename.endswith('.safetensors') for file in model.siblings):
                variants["safetensors"].append(model.modelId)
    
    return variants

def print_model_uris(variants):
    for format, repos in variants.items():
        print(f"\n{format.capitalize()} variants:")
        if repos:
            for repo in repos:
                print(f"https://huggingface.co/{repo}")
        else:
            print("No variants found")

def main():
    # Get the base model name from user input with a default value
    default_model = "microsoft/phi-3.5-mini-instruct"
    base_model_name = input(f"Enter the base model name (default: {default_model}): ").strip() or default_model

    print(f"\nSearching for variants of {base_model_name}...")
    variants = find_model_variants(base_model_name)
    
    print("\nFound the following model variants:")
    for format, repos in variants.items():
        print(f"\n{format.capitalize()} variants:")
        if repos:
            for i, repo in enumerate(repos, 1):
                print(f"  {i}. {repo}")
        else:
            print("  No variants found")

    print_model_uris(variants)

if __name__ == "__main__":
    main()