

import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn.functional as F

# ======================================================
# 1. Preparation
# ======================================================

# Choose device and CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ViT-B/32"  # or "RN50"
model, preprocess = clip.load(model_name, device=device)

# Example image paths (replace with your own images if needed)
image_paths = ["cat.jpg", "dog.png", "giraffe.jpg"]

# Corresponding text prompts
text_prompts = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a giraffe"
]

# Load and preprocess images
images = [preprocess(Image.open(p)) for p in image_paths]
image_input = torch.stack(images).to(device)

# Tokenize text
text_tokens = clip.tokenize(text_prompts).to(device)

# ======================================================
# 2. Encode images and texts
# ======================================================

with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_tokens)

# Normalize (L2 normalization)
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features  /= text_features.norm(dim=-1, keepdim=True)

# Confirm shapes and unit length
print(f"Image features shape: {image_features.shape}")
print(f"Text features shape:  {text_features.shape}")
print("Image feature norms:", image_features.norm(dim=-1))
print("Text feature norms: ", text_features.norm(dim=-1))

# ======================================================
# 3. Compute cosine similarities and visualize
# ======================================================

# Compute similarity matrix (cosine similarity)
similarity = image_features @ text_features.T  # shape (n_images, n_texts)

# Convert to numpy for plotting
sim_matrix = similarity.cpu().numpy()

# Print similarity matrix
print("\nCosine similarity matrix:\n", sim_matrix)

# Identify best-matching text prompt for each image
best_matches = sim_matrix.argmax(axis=1)
for i, idx in enumerate(best_matches):
    print(f"Image {i+1} ({image_paths[i]}) best matches prompt: '{text_prompts[idx]}'")

# Plot heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(sim_matrix, annot=True, cmap="viridis",
            xticklabels=text_prompts,
            yticklabels=[p.split('.')[0] for p in image_paths],
            fmt=".2f")
plt.title(f"CLIP ({model_name}) Cosine Similarity")
plt.xlabel("Text prompts")
plt.ylabel("Images")
plt.tight_layout()
plt.show()
#
# Image features shape: torch.Size([3, 512])
# Text features shape:  torch.Size([3, 512])
# Image feature norms: tensor([1., 1., 1.])
# Text feature norms:  tensor([1.0000, 1.0000, 1.0000])
#
# Cosine similarity matrix:
#  [[0.28801423 0.23099326 0.19604284]
#  [0.19946209 0.27622324 0.19570883]
#  [0.19745588 0.2047004  0.31226808]]
# Image 1 (cat.jpg) best matches prompt: 'a photo of a cat'
# Image 2 (dog.png) best matches prompt: 'a photo of a dog'
# Image 3 (giraffe.jpg) best matches prompt: 'a photo of a giraffe'

# ======================================================
# 4. EXTRA: InfoNCE loss experiment
# ======================================================
#
# def clip_infonce_loss(image_emb, text_emb, tau):
#     """Compute symmetric InfoNCE loss for given temperature tau."""
#     logits = (image_emb @ text_emb.T) / tau
#     labels = torch.arange(logits.shape[0], device=logits.device)
#     loss_i = F.cross_entropy(logits, labels)
#     loss_t = F.cross_entropy(logits.T, labels)
#     return 0.5 * (loss_i + loss_t)
#
# # Try multiple temperature values
# temperatures = [0.07, 0.5, 1.0, 2.0, 5.0]
# loss_values = []
#
# for tau in temperatures:
#     loss = clip_infonce_loss(image_features, text_features, tau)
#     loss_values.append(loss.item())
#     print(f"τ = {tau:.2f} → InfoNCE loss = {loss.item():.4f}")
#
# # Plot loss vs. temperature
# plt.figure(figsize=(6, 4))
# plt.plot(temperatures, loss_values, marker='o')
# plt.title("Effect of Temperature on InfoNCE Loss")
# plt.xlabel("Temperature (τ)")
# plt.ylabel("Loss value")
# plt.grid(True)
# plt.show()
