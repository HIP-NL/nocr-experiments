# Vision Language Model Experiments

This repository contains experiments evaluating Google's Gemini Flash models for extracting structured data from historical handwritten Dutch tax records from Utrecht, 1899.

## Overview

We test different Gemini Flash model variants on their ability to transcribe handwritten tax records into structured JSON format. The experiments compare:

- **Models**: gemini-2.0-flash, gemini-2.5-flash, gemini-3.0-flash (plus lite variants)
- **Strategies**: Few-shot learning (with 3 examples) vs. zero-shot learning
- **Thinking budgets**: 0 tokens (no thinking) vs. 2000 tokens (with chain-of-thought reasoning)

## 🚀 Quick Start

### Running on Google Colab

The easiest way to explore this project is via the notebook on [Google Colab](https://githubtocolab.com/HIP-NL/nocr-experiments/blob/main/notebook/vlm_extraction_demo.ipynb)

### Running Locally

#### Prerequisites

- Python 3.11+
- R 4.0+ with packages: `data.table`, `jsonlite`, `stringdist`, `tinyplot`
- Google Gemini API key

#### Setup

```bash
# Clone the repository
git clone https://github.com/HIP-NL/nocr-experiments.git
cd nocr-experiments


# Install Python dependencies
pip install -r experiments/requirements.txt

# Set your API key
export GEMINI_API_KEY="your-api-key-here"
```

#### Run Experiments

```bash
cd experiments
python run_experiments.py
```

This will generate predictions and metadata in the `results/` directory.

#### Evaluate Results

```bash
cd evaluation
Rscript evaluate.R
```

This will compute error metrics and generate plots in `evaluation/figures/`.

##  Data

### Input Images

Historical scans of tax records with 5 columns:
- Sequential number (volgnummer)
- Name (title, initials, surname, maiden name)
- Address (street, house number)
- Tax class
- Tax amount

### Ground Truth Format

Each record is a JSON array of objects:

```json
[
  {
    "volgnummer": 1861,
    "title": null,
    "initials": "G.",
    "surname": "Fukkink",
    "maiden_name": null,
    "street": "Oudekamp",
    "house_number": "10",
    "class": 2,
    "tax": 4.5
  },
  ...
]
```

### Output File Naming

All result files follow the pattern:

```
{image_name}__{model}__{strategy}__{thinking}.json
```

Example:  `NL-UtHUA_A376076_000033_l__gemini-2.5-flash-lite__fewshot__thinking2000.json`

## 📈 Evaluation Metrics

- **Character Error Rate (CER)**: Levenshtein distance between predicted and ground truth text, normalized by total characters
- **Cell Error Rate**: Proportion of fields (cells) with any error
- **Cost**: Estimated API cost in USD based on token usage and model pricing

## Key Findings

- VLMs directly understand both the text and the document structure, and can directly return structured data, drastically reducing post-processing efforts.
- Few-shot learning consistently outperforms zero-shot
- Thinking budgets provide only modest improvements for most models

## 📄 License

GPL-3 

## Acknowledgments

- Utrecht City Archives for providing access to historical records
- Google for the Gemini API
