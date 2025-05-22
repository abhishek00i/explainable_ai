# XAI for NLI Model - Proof-of-Concept Report

## 1. Objective

This report details the findings from a proof-of-concept (PoC) project aimed at exploring and implementing explainable AI (XAI) techniques for a Natural Language Inference (NLI) model. The primary goal was to understand how attention mechanisms and local surrogate models (LIME) can provide insights into the NLI model's decision-making process.

## 2. System Architecture and Components

The system consists of the following key components:

*   **NLI Model**: A DistilBERT model fine-tuned for sequence classification on a minimal NLI task.
    *   *Implementation*: [01_nli_model_poc.ipynb](../notebooks/01_nli_model_poc.ipynb)
*   **Attention Visualization**: Techniques to extract and visualize attention weights from the transformer model.
    *   *Implementation*: [02_attention_visualization_poc.ipynb](../notebooks/02_attention_visualization_poc.ipynb)
*   **LIME (Local Interpretable Model-agnostic Explanations)**: Application of LIME to explain individual predictions of the NLI model.
    *   *Implementation*: [03_lime_explanation_poc.ipynb](../notebooks/03_lime_explanation_poc.ipynb)
*   **Interactive Explainer Notebook:** A Jupyter notebook that allows users to input a custom premise and hypothesis, receive an NLI prediction, and view both Attention and LIME explanations.
    *   Located in: `notebooks/04_interactive_nli_explainer.ipynb`
*   **Code Utilities:** Common functions for LIME predictor setup and attention processing/plotting have been refactored into `src/xai_utils.py` for better code organization and reusability across notebooks.

The project directory is structured to separate source code (`src/`), data (`data/` - conceptual for this PoC), notebooks (`notebooks/`), and this report (`report/`).

## 3. Findings from XAI Techniques

### 3.1. Attention Visualization

The attention visualization notebook ([02_attention_visualization_poc.ipynb](../notebooks/02_attention_visualization_poc.ipynb)) provided insights into how the model weighs different tokens when processing premise-hypothesis pairs.

**Example Input:**
*   Premise: "A man is playing a guitar."
*   Hypothesis: "A person is making music."
*   Predicted Label by Model: Entailment

**Overall Attention Heatmap (Averaged Heads, Last Layer):**
*   **Strong Self-Attention:** Tokens like `[CLS]` and `[SEP]` showed high attention to themselves, as expected due to their roles in sequence representation and separation.
*   **Premise-Hypothesis Interaction:**
    *   The token "guitar" in the premise showed noticeable attention towards "music" in the hypothesis.
    *   "man" in the premise attended to "person" in the hypothesis.
    *   Conversely, "music" and "person" in the hypothesis also attended back to "guitar" and "man" respectively.
*   **Contextual Word Grouping:** Within the premise, "playing" and "guitar" showed mutual attention. Within the hypothesis, "making" and "music" also showed mutual attention.
*   **Special Tokens:** The `[CLS]` token, used for classification, appeared to aggregate information broadly from both premise and hypothesis, with distributed attention scores across meaningful words like "man", "guitar", "person", "music".

**Separated Attention Views (Premise-Only & Hypothesis-Only Self-Attention):**
*   **Premise Self-Attention ("A man is playing a guitar.")**:
    *   "man" attended to "playing" and "guitar".
    *   "playing" attended strongly to "guitar" and "man".
    *   "guitar" attended back to "playing" and "man".
    This shows the model successfully identifying the core action and entities within the premise.
*   **Hypothesis Self-Attention ("A person is making music.")**:
    *   "person" attended to "making" and "music".
    *   "making" attended to "music" and "person".
    *   "music" attended to "making" and "person".
    Similar to the premise, this view highlighted the model's ability to form a coherent understanding of the hypothesis statement independently.

**Interpretation:**
Attention maps suggest that the model learns to focus on semantically related words across the premise and hypothesis (e.g., "guitar" <-> "music", "man" <-> "person") and also builds internal representations of each sentence by focusing on key relationships within them. The `[CLS]` token effectively gathers these relational signals. The patterns observed are generally intuitive and align with how a human might assess entailment for this example.

### 3.2. LIME (Local Interpretable Model-agnostic Explanations)

LIME was used to explain the same prediction (Entailment for "A man is playing a guitar. [SEP] A person is making music.") in the [03_lime_explanation_poc.ipynb](../notebooks/03_lime_explanation_poc.ipynb) notebook.

**Key Findings from LIME:**
*   **Words Supporting Entailment (Positive Contribution):**
    *   "music" (from hypothesis)
    *   "guitar" (from premise)
    *   "playing" (from premise)
*   **Words Against Entailment (or supporting other labels - Negative Contribution, if any were prominent):**
    *   For this specific entailment example, most of the key premise/hypothesis words were shown as supporting the entailment. LIME might highlight words that, if absent, would weaken the entailment prediction or, if present, might push towards contradiction/neutral if the relationship was different.
*   **The `[SEP]` token** sometimes appears with a weight, indicating LIME considers its presence (or absence in perturbed samples) as a feature. Its role is structural for the model.

**Interpretation:**
LIME identified "music", "guitar", and "playing" as the primary drivers for the entailment prediction. This aligns with human intuition, as the act of "playing a guitar" strongly implies "making music". LIME provides a word-level attribution, which is complementary to the token-level, more structural view offered by attention. The results from LIME are consistent with the attention patterns, where these words also showed significant cross-attention.

## 4. Limitations and Future Work

This PoC has several limitations and opens avenues for future work:

*   **Minimal Dataset:** The NLI model was trained on a very small, illustrative dataset. A production model would require training on a standard NLI benchmark (e.g., SNLI, MNLI).
*   **Basic XAI Implementation:**
    *   **Attention:** Only averaged attention from the last layer was deeply analyzed. Exploring individual heads, different layers, or attention flow could yield richer insights.
    *   **LIME:** `num_samples` was kept relatively low for speed. Higher values would provide more stable explanations. The `bow=False` setting is an approximation; more sophisticated perturbation strategies for transformers could be explored.
*   **Limited Scope of XAI Techniques:** Other XAI methods like SHAP (KernelSHAP or PartitionSHAP for text), Integrated Gradients, TCAV, or neuron activation analysis were not implemented but are listed conceptually in `02_attention_visualization_poc.ipynb`.
*   **No Human Evaluation:** The interpretability of the explanations was not formally evaluated with human users.
*   **Single Model Architecture:** Only DistilBERT was explored. Comparing explanations across different model architectures could be insightful.

**Future Work:**
1.  **Train on Standard NLI Benchmark:** Improve model robustness and generalizability.
2.  **Expand XAI Toolkit:** Implement and compare other XAI techniques (SHAP, Integrated Gradients).
3.  **In-depth Attention Analysis:** Systematically study individual attention heads and layer-wise attention patterns.
4.  **Quantitative Evaluation of Explanations:** Explore metrics for faithfulness and stability of explanations.
5.  **User Studies:** Conduct studies to assess the helpfulness and clarity of the generated explanations for different user groups.
6.  **Explore Counterfactual Explanations:** Generate examples of what minimal changes to an input would alter the model's prediction.

## 5. How to Run the Notebooks

1.  Ensure Python environment is set up with dependencies from `explainable_nli_system/requirements.txt`.
2.  It's recommended to run `notebooks/01_nli_model_poc.ipynb` first if you want to regenerate the model. Otherwise, the pre-saved model in `src/nli_model/` will be used by other notebooks.
3.  To see individual XAI techniques demonstrated:
    *   `notebooks/02_attention_visualization_poc.ipynb` (demonstrates attention visualization, uses utility functions)
    *   `notebooks/03_lime_explanation_poc.ipynb` (demonstrates LIME, uses utility functions)
4.  **For interactive explanations with custom input:**
    *   Run `notebooks/04_interactive_nli_explainer.ipynb`. This notebook loads the model and allows you to input a premise/hypothesis to see both Attention and LIME explanations.
