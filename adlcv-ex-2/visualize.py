import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch

def visualize_attention(model, img_tensor, label, patch_size):
    model.eval()
    with torch.no_grad():
        out = model(img_tensor)

    attentions = model.transformer_blocks[-1].last_attention_weights
    
    cls_attn = attentions[:, 0, 1:] 
    cls_attn = cls_attn.mean(dim=0) 
    
    w = h = int(cls_attn.shape[0]**0.5)
    cls_attn = cls_attn.reshape(h, w).cpu().numpy()
    
    img_size = img_tensor.shape[-1]
    cls_attn = cv2.resize(cls_attn / cls_attn.max(), (img_size, img_size))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_tensor[0].permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(img_tensor[0].permute(1, 2, 0).cpu().numpy()) # Show image
    axes[1].imshow(cls_attn, cmap='jet', alpha=0.6) # Overlay heatmap
    axes[1].set_title("Attention Heatmap Overlay")
    axes[1].axis('off')

    plt.title(f"Attention Visualization (correctly guess = {label[0].item() == out.argmax(dim=1)[0].item()})")

    plt.tight_layout()
    plt.show()