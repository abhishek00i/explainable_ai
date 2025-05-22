# Explainable AI System for NLI Models - Proof-of-Concept

This project demonstrates a proof-of-concept (PoC) system for applying and visualizing eXplainable AI (XAI) techniques on a Natural Language Inference (NLI) model. The goal is to provide insights into the model's decision-making process using methods like attention visualization and LIME.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Directory Structure](#directory-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage - Running the Notebooks](#usage---running-the-notebooks)
- [XAI Techniques Implemented](#xai-techniques-implemented)
- [Future Work](#future-work)
- [Report](#report)

## Project Overview

Natural Language Inference (NLI) is the task of determining whether a "hypothesis" is true (entailment), false (contradiction), or undetermined (neutral) given a "premise". While deep learning models, particularly transformers, have achieved high accuracy on NLI tasks, their internal workings are often opaque. This project aims to shed light on these "black boxes" by:

1.  Fine-tuning a DistilBERT model for NLI on a minimal dataset.
2.  Implementing attention visualization to understand which parts of the input text the model focuses on.
3.  Applying LIME (Local Interpretable Model-agnostic Explanations) to identify key words that contribute to a specific prediction.
4.  Providing an interactive notebook to explore explanations for custom premise-hypothesis pairs.
5.  Refactoring common XAI functionalities into reusable utility functions.

## Features

*   **NLI Model Training PoC**: A Jupyter notebook (`01_nli_model_poc.ipynb`) to fine-tune a DistilBERT model for NLI and save it.
*   **Attention Visualization**:
    *   Visualizes attention weights from the transformer model's last layer (averaged across heads).
    *   Shows full attention (premise & hypothesis combined) and separated self-attention for premise and hypothesis.
    *   Implementation in `02_attention_visualization_poc.ipynb`.
*   **LIME Explanations**:
    *   Applies LIME to generate word-level importance scores for individual NLI predictions.
    *   Implementation in `03_lime_explanation_poc.ipynb`.
*   **Interactive Explainer**:
    *   A notebook (`04_interactive_nli_explainer.ipynb`) allowing users to input their own premise and hypothesis.
    *   Displays the model's prediction, attention heatmaps, and LIME feature importance scores for the custom input.
*   **Utility Functions**: Common code for LIME predictor setup and attention processing/plotting is centralized in `src/xai_utils.py`.

## Directory Structure

```
explainable_nli_system/
├── notebooks/                  # Jupyter notebooks for different PoC stages
│   ├── 01_nli_model_poc.ipynb
│   ├── 02_attention_visualization_poc.ipynb
│   ├── 03_lime_explanation_poc.ipynb
│   └── 04_interactive_nli_explainer.ipynb
├── report/                     # Project report
│   └── REPORT.md
├── src/                        # Source code
│   ├── nli_model/              # Saved fine-tuned NLI model and tokenizer
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   └── vocab.txt
│   └── xai_utils.py            # Utility functions for XAI
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Setup and Installation

1.  **Clone the repository (if applicable).**
2.  **Create a Python virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use ` .venv\Scripts\activate `
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: It's recommended to install `torch` with CUDA support if you have a compatible GPU. Follow instructions from the [PyTorch website](https://pytorch.org/) for your specific OS and CUDA version, then install other packages from `requirements.txt`.*

## Usage - Running the Notebooks

It's recommended to run the notebooks from the `explainable_nli_system/notebooks/` directory.

1.  **Start Jupyter Lab or Jupyter Notebook:**
    ```bash
    jupyter lab
    # or
    jupyter notebook
    ```

2.  **Run notebooks in the following order for a full workflow:**

    *   **`01_nli_model_poc.ipynb`**:
        *   This notebook fine-tunes a DistilBERT model on a sample NLI dataset and saves the model and tokenizer to `src/nli_model/`.
        *   Run this first if you want to re-train the model. If you skip this, the other notebooks will use the pre-saved model in `src/nli_model/`.

    *   **`02_attention_visualization_poc.ipynb`**:
        *   Loads the fine-tuned model.
        *   Demonstrates how to extract and visualize attention weights.
        *   Uses utility functions from `src/xai_utils.py`.

    *   **`03_lime_explanation_poc.ipynb`**:
        *   Loads the fine-tuned model.
        *   Applies LIME to explain a sample prediction.
        *   Uses utility functions from `src/xai_utils.py`.

    *   **`04_interactive_nli_explainer.ipynb`**:
        *   **This is the main interactive notebook.**
        *   Loads the fine-tuned model.
        *   Prompts the user to enter a premise and a hypothesis.
        *   Displays the model's NLI prediction.
        *   Visualizes attention patterns (full, premise-only, hypothesis-only).
        *   Shows LIME explanations for the prediction.
        *   Uses utility functions from `src/xai_utils.py`.

## XAI Techniques Implemented

*   **Attention Visualization**: Transformer models use attention mechanisms to weigh the importance of different tokens when processing input. Visualizing these attention scores can help understand which parts of the premise and hypothesis the model focuses on when making a prediction. We visualize the last layer's attention, averaged across heads.
*   **LIME (Local Interpretable Model-agnostic Explanations)**: LIME explains individual predictions of any black-box model by learning a simpler, interpretable model (e.g., linear regression) locally around the prediction. For text, LIME perturbs the input (e.g., by removing words) and sees how predictions change to assign importance scores to words.

## Future Work

This PoC can be extended in several ways:

*   Train on a standard NLI benchmark (e.g., SNLI, MNLI) for a more robust model.
*   Implement other XAI techniques like SHAP, Integrated Gradients, or TCAV.
*   Conduct a more in-depth analysis of attention (e.g., individual heads, layer-wise patterns).
*   Perform quantitative evaluations of explanation faithfulness and stability.
*   Conduct user studies to assess the helpfulness of the explanations.

## Report

A detailed report summarizing the project's objectives, architecture, findings, and limitations can be found in [report/REPORT.md](report/REPORT.md).
