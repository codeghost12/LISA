# LISA: Linguistic and Statistical Analysis for AI-Generated Text Detection

[![Conference](https://img.shields.io/badge/Conference-ICMLA%202025-1f6feb)](https://ieeexplore.ieee.org/document/11471380)
[![Paper](https://img.shields.io/badge/Paper-IEEE%20Xplore-blue)](https://ieeexplore.ieee.org/document/11471380)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FICMLA66185.2025.00030-orange)](https://doi.org/10.1109/ICMLA66185.2025.00030)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-torch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-FFD21E)](https://huggingface.co/docs/transformers/index)
[![spaCy](https://img.shields.io/badge/spaCy-NLP-09A3D5)](https://spacy.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E)](https://scikit-learn.org/stable/)

📄 **Paper:** [IEEE Xplore](https://ieeexplore.ieee.org/document/11471380) | [DOI](https://doi.org/10.1109/ICMLA66185.2025.00030)

> 🎉🎉🎉 **Our paper has been accepted at ICMLA 2025!**

Official repository for the paper:

**Vaishnavi Sen and Rashida Hasan.**  
**"A Feature-based Linguistic and Statistical Framework for AI-Generated Text Detection."**  
*2025 International Conference on Machine Learning and Applications (ICMLA)*, pp. 184-191.  
IEEE Xplore: https://ieeexplore.ieee.org/document/11471380  
DOI: https://doi.org/10.1109/ICMLA66185.2025.00030

## Overview

LISA is a feature-based framework for distinguishing human-written text from AI-generated text using interpretable linguistic and statistical signals. Instead of relying only on end-to-end deep classifiers, LISA combines handcrafted features such as:

- lexical diversity
- repetition statistics
- entropy
- part-of-speech distributions
- parse depth
- next-sentence coherence
- topic-distribution features
- bag-of-words / count-based text features

The repository also includes baseline implementations for BERT, RoBERTa, GPTZero-style perplexity scoring, and DetectGPT-style perturbation scoring.

## Core Packages

- `torch`
- `transformers`
- `spacy`
- `scikit-learn`
- `xgboost`
- `nltk`
- `pandas`
- `numpy`

## Repository Layout

```text
.
├── data/
│   ├── hc3.csv
│   └── xsum.csv
├── lisa/
│   └── lisa_models.py
├── sota/
│   ├── bert.py
│   ├── detectgpt.py
│   ├── gptzero.py
│   └── roberta.py
├── feature_extraction.py
├── main.py
└── requirements.txt
```

## Requirements

- Python 3.10+ recommended
- `pip`
- enough disk space for Hugging Face model downloads
- CPU is sufficient for small experiments, but GPU is strongly recommended for transformer baselines

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

Notes:

- The first run will also download NLTK resources and pretrained Hugging Face checkpoints.
- The baseline models are substantially more expensive than the classical LISA models.

## Data Format

The bundled CSV files use the columns:

- `texts`
- `labels`

The runner also supports datasets that use:

- `text`
- `label`

Expected label convention:

- `0` = human-written
- `1` = AI-generated

## Usage

Run the default experiment on XSum:

```bash
python3 main.py --dataset data/xsum.csv
```

Run the same pipeline on HC3:

```bash
python3 main.py --dataset data/hc3.csv
```

Run only the LISA models and skip the heavyweight baselines:

```bash
python3 main.py --dataset data/xsum.csv --skip-sota
```

Run a lighter sanity-check experiment:

```bash
python3 main.py --dataset data/xsum.csv --skip-sota --skip-ratio-evals --subset-sizes 10000
```

Available CLI options:

```bash
python3 main.py --help
```

## What `main.py` Does

The main entry point:

1. loads a CSV dataset
2. resolves the text and label columns
3. extracts LISA features for all documents
4. evaluates the included SOTA baselines
5. evaluates the LISA classifiers on the full dataset and subset experiments
6. optionally evaluates class-ratio experiments

## Reproducibility Notes

- The transformer baselines download models from Hugging Face at runtime.
- DetectGPT-style and GPTZero-style scoring can be slow on large datasets.
- The included code is designed for research experimentation and comparison, not production deployment.

## Citation

If you use this repository, please cite the paper:

```bibtex
@inproceedings{sen2025lisa,
  author    = {Sen, Vaishnavi and Hasan, Rashida},
  title     = {A Feature-based Linguistic and Statistical Framework for AI-Generated Text Detection},
  booktitle = {2025 International Conference on Machine Learning and Applications (ICMLA)},
  pages     = {184--191},
  year      = {2025},
  doi       = {10.1109/ICMLA66185.2025.00030},
  url       = {https://doi.org/10.1109/ICMLA66185.2025.00030}
}
```

## Acknowledgments

This repository includes experiments built around the XSum and HC3 datasets, along with baseline implementations inspired by prior work on AI-generated text detection.
