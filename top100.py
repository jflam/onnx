from huggingface_hub import HfApi
from pprint import pprint
from datetime import datetime, timedelta
import re

def parse_date(date_obj):
    if isinstance(date_obj, datetime):
        return date_obj
    if not date_obj:
        print(f"Warning: Empty date value. Using current date instead.")
        return datetime.now(datetime.now().astimezone().tzinfo)
    try:
        if isinstance(date_obj, str):
            return datetime.fromisoformat(date_obj.replace('Z', '+00:00'))
        else:
            raise ValueError("Unexpected date format")
    except ValueError:
        print(f"Warning: Unable to parse date '{date_obj}'. Using current date instead.")
        return datetime.now(datetime.now().astimezone().tzinfo)

def extract_base_model_name(model_id):
    # Known base models and their variants
    known_models = {
        'llama': ['llama', 'llava', 'meta-llama'],
        'mistral': ['mistral', 'mathstral'],
        'gemma': ['gemma'],
        'qwen': ['qwen'],
        'florence': ['florence'],
        'flux': ['flux'],
        'phi': ['phi'],
        'stable-diffusion': ['stable-diffusion', 'stable'],
        'vram': ['vram'],
        'opensora': ['opensora'],
        'dolphin': ['dolphin'],
        'yi': ['yi'],
        'depth': ['depth-anything'],
        'midnight': ['midnight'],
        'miqu': ['miqu'],
        'internlm': ['internlm'],
        'vits': ['vits'],
        't5': ['t5'],
        'mixtral': ['mixtral'],
    }
    
    # Check for known base models in the full model_id
    lower_model_id = model_id.lower()
    for base_model, variants in known_models.items():
        if any(variant in lower_model_id for variant in variants):
            return base_model.capitalize()
    
    # Remove version numbers and common suffixes
    name = re.sub(r'[-_]v\d+(\.\d+)*', '', lower_model_id)
    name = re.sub(r'[-_](instruct|gguf|awq|gptq|bnb-4bit|fp8|hf|quantized|diffusers|tokenizer).*$', '', name)
    
    # Remove size indicators and common words
    name = re.sub(r'[-_]?\d+[bm]', '', name)
    name = re.sub(r'[-_](mini|small|medium|large|base|xxl|reward)$', '', name)
    
    # Split by slash and take the last part
    parts = name.split('/')
    name = parts[-1]
    
    # Remove common prefixes
    name = re.sub(r'^(tiny-dummy-|dummy-)', '', name)
    
    # Split by underscore or hyphen and take the first part
    name = re.split(r'[_-]', name)[0]
    
    # Remove numbers and hyphens at the end
    name = re.sub(r'[-_]\d+$', '', name)
    name = name.rstrip('-_')
    
    # Limit the length of the name
    name = name[:10]
    
    # Capitalize the first letter
    name = name.capitalize()
    
    return name if len(name) > 1 else "Unknown"

# The rest of the script remains the same
# The rest of the script remains the same

def get_top_recent_models(limit=100, months=3):
    api = HfApi()
    
    models = api.list_models(sort="downloads", direction=-1, limit=1000, full=True)
    
    three_months_ago = datetime.now(datetime.now().astimezone().tzinfo) - timedelta(days=30*months)
    
    recent_models = []
    for model in models:
        creation_date = parse_date(getattr(model, 'createdAt', None) or getattr(model, 'created_at', None))
        if creation_date > three_months_ago:
            recent_models.append(model)
    
    recent_models.sort(key=lambda x: getattr(x, 'downloads', 0), reverse=True)
    
    return recent_models[:limit]

def main():
    top_models = get_top_recent_models()
    
    print(f"Top {len(top_models)} models on Hugging Face Hub in the last 3 months:")
    print("-" * 70)
    
    for i, model in enumerate(top_models, 1):
        created_at = parse_date(getattr(model, 'createdAt', None) or getattr(model, 'created_at', None))
        downloads = getattr(model, 'downloads', 'N/A')
        base_name = extract_base_model_name(model.modelId)
        print(f"{i}. {model.modelId}")
        print(f"   URI: https://huggingface.co/{model.modelId}")
        print(f"   Created: {created_at.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"   Downloads: {downloads:,}" if isinstance(downloads, int) else f"   Downloads: {downloads}")
        print(f"   Predicted Base Model: {base_name}")
        print()
    
    print("\nExample of a complete model object:")
    print("-" * 50)
    pprint(vars(top_models[0]))

if __name__ == "__main__":
    main()