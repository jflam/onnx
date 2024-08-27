# Hugging Face Model Analysis Scripts

This repository contains two Python scripts for analyzing and visualizing download statistics for various Hugging Face model variants.

## Files

1. `app.py`: Main script for fetching and analyzing model data from Hugging Face.
2. `chart.py`: Script for creating a bar chart visualization of the analysis results.

## Prerequisites

- Python 3.x
- Hugging Face API token (set as an environment variable `HF_TOKEN`)

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```
   pip install huggingface_hub tenacity matplotlib numpy
   ```

3. Set up your Hugging Face API token as an environment variable:
   ```
   export HF_TOKEN=your_token_here
   ```

## Usage

### Running the analysis script (app.py)

```
python app.py [--model MODEL_NAME] [--csv]
```

Options:
- `--model MODEL_NAME`: Analyze a specific model (optional)
- `--csv`: Output results in CSV format

If no model is specified, the script will analyze a predefined list of popular models.

### Generating the chart (chart.py)

After running `app.py` with the `--csv` option:

```
python chart.py
```

This will generate a bar chart (`chart.png`) visualizing the download statistics for different model variants.

## Output

- Console output with model variant counts and download statistics
- CSV file (`model_analysis_results.csv`) with detailed results (if `--csv` option is used)
- PNG image (`chart.png`) with a bar chart visualization of the results

## Default Models

The script analyzes the following models by default:

- Microsoft Phi models
- Google Gemma models
- Meta Llama models
- Mistral AI models
- Qwen models

You can modify the `DEFAULT_MODELS` list in `app.py` to change this selection.

## Error Handling

The script implements retry logic for API calls to handle potential network issues or rate limiting.

## Contributing

Feel free to fork this repository and submit pull requests with any improvements or additional features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.