import torch
import numpy # numpy is not explicitly used in get_lime_predictor, but good for a utils file
import matplotlib.pyplot as plt
import seaborn as sns # Renamed from 'seaborn' to 'sns' for conventional import alias

def get_lime_predictor(model, tokenizer, device):
    """
    Returns a predictor function compatible with LIME.
    """
    def predictor(texts):
        # Ensure model is in eval mode for predictions
        model.eval()
        # LIME sometimes passes a single string, ensure it's a list for tokenizer
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
    return predictor

def process_model_attentions(attentions_output, input_ids_tensor, tokenizer):
    """
    Processes model's attention output to get averaged attentions and tokens.
    Args:
        attentions_output: Tuple of attention tensors from model output (e.g., model.outputs.attentions).
        input_ids_tensor: Tensor of input_ids (e.g., inputs['input_ids']).
        tokenizer: The tokenizer instance.
    Returns:
        Tuple of (numpy.ndarray, list): Averaged attention weights (seq_len, seq_len) and tokens.
    """
    # For DistilBERT, attentions_output is a tuple of tensors, one for each layer.
    # Each tensor is of shape (batch_size, num_heads, sequence_length, sequence_length).
    last_layer_attentions = attentions_output[-1] 
    
    # Average over heads. Assuming batch_size=1 for explanation, so take the first item [0].
    avg_attentions = last_layer_attentions.mean(dim=1)[0] 
    avg_attentions_np = avg_attentions.cpu().detach().numpy()
    
    # Convert input_ids to tokens
    # Ensure input_ids_tensor is on CPU and converted to numpy for tokenization
    tokens = tokenizer.convert_ids_to_tokens(input_ids_tensor[0].cpu().numpy())
    return avg_attentions_np, tokens

def plot_heatmap(data, x_labels, y_labels, title="Attention Heatmap", figsize_scale_x=1.2, figsize_scale_y=1.5, min_size_x=10, min_size_y=8):
    """
    Plots a heatmap using seaborn.
    Args:
        data (numpy.ndarray): The 2D data to plot.
        x_labels (list): Labels for the x-axis.
        y_labels (list): Labels for the y-axis.
        title (str): Title of the plot.
        figsize_scale_x (float): Factor to scale heatmap width by number of x_labels.
        figsize_scale_y (float): Factor to scale heatmap height by number of y_labels.
        min_size_x (int): Minimum width of the figure.
        min_size_y (int): Minimum height of the figure.
    """
    plt.figure(figsize=(max(min_size_x, len(x_labels) / figsize_scale_x), 
                       max(min_size_y, len(y_labels) / figsize_scale_y)))
    sns.heatmap(data, xticklabels=x_labels, yticklabels=y_labels, cmap='viridis', annot=False)
    plt.title(title)
    plt.xlabel("Attended-To Tokens")
    plt.ylabel("Attending Tokens")
    plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability
    plt.yticks(rotation=0) # Keep y-axis labels horizontal
    plt.tight_layout() # Adjust layout to prevent labels from being cut off
    plt.show()
