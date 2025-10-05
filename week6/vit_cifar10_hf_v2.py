"""
============================================================
Vision Transformer (ViT) Fine-Tuning and Comparison on CIFAR-10
============================================================
"""

# -----------------------------
# Imports
# -----------------------------
from datasets import load_dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    ViTConfig,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

# -----------------------------
# Custom Dataset Class to handle CIFAR-10 properly
# -----------------------------
class CIFAR10Dataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['img']
        label = item['label']

        # Process the image
        inputs = self.processor(images=image, return_tensors="pt")

        return {
            'pixel_values': inputs['pixel_values'].squeeze(),
            'labels': torch.tensor(label)
        }

# -----------------------------
# (a) Load and preprocess CIFAR-10
# -----------------------------
print("Loading CIFAR-10 dataset...")

train_ds, test_ds = load_dataset("cifar10", split=["train", "test"])
splits = train_ds.train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = splits["train"], splits["test"]

print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

# Image processor for resizing and normalization
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)

# Create custom datasets
train_dataset = CIFAR10Dataset(train_ds, processor)
val_dataset = CIFAR10Dataset(val_ds, processor)
test_dataset = CIFAR10Dataset(test_ds, processor)

# Test the dataset
print("Testing dataset...")
sample = train_dataset[0]
print("Sample keys:", list(sample.keys()))
print("Pixel values shape:", sample['pixel_values'].shape)
print("Label:", sample['labels'])

# -----------------------------
# (b) Load pretrained model
# -----------------------------
print("\nLoading pretrained ViT model...")

model_pretrained = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=10,
    ignore_mismatched_sizes=True
)

# -----------------------------
# (c) Fine-tuning setup
# -----------------------------
# Freeze all transformer encoder parameters
for name, param in model_pretrained.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False

print("Backbone frozen. Only classifier layer will be fine-tuned.")

# Count parameters
trainable_params = sum(p.numel() for p in model_pretrained.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model_pretrained.parameters())
print(f"Trainable parameters: {trainable_params:,}/{total_params:,} ({100*trainable_params/total_params:.2f}%)")

# -----------------------------
# (d) Training configuration
# -----------------------------
def compute_metrics(eval_pred):
    """Compute classification accuracy."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


#args_pretrained_fixed = TrainingArguments(
    learning_rate=5e-6,      # Much smaller for fine-tuning
    max_grad_norm=1.0,         # Prevent gradient explosion
    fp16=False,                # More stability
    # ... other parameters
#)

args_pretrained = TrainingArguments(
    output_dir="./results_pretrained",
    evaluation_strategy="epoch",  # CHANGED: eval_strategy -> evaluation_strategy
    save_strategy="epoch",
    learning_rate=5e-6, # smaller for fine tuning
    max_grad_norm=1,
    fp16=False,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=100,
    logging_dir="./logs_pretrained",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to=None,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
)

trainer_pretrained = Trainer(
    model=model_pretrained,
    args=args_pretrained,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# -----------------------------
# Train pretrained model
# -----------------------------
print("\nTraining pretrained ViT (fine-tuning)...")
trainer_pretrained.train()

metrics_pretrained = trainer_pretrained.evaluate(test_dataset)
print("\nPretrained model test accuracy:", metrics_pretrained["eval_accuracy"])

# -----------------------------
# Save model and logs
# -----------------------------
trainer_pretrained.save_model("./vit_cifar10_finetuned")
processor.save_pretrained("./vit_cifar10_finetuned")

# -----------------------------
# (e) Train from scratch for comparison
# -----------------------------
print("\nTraining ViT from scratch for comparison...")

config = ViTConfig.from_pretrained(model_name, num_labels=10)
model_scratch = ViTForImageClassification(config)

args_scratch = TrainingArguments(
    output_dir="./results_scratch",
    evaluation_strategy="epoch",  # CHANGED: eval_strategy -> evaluation_strategy
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs_scratch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to=None,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
)

trainer_scratch = Trainer(
    model=model_scratch,
    args=args_scratch,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer_scratch.train()
metrics_scratch = trainer_scratch.evaluate(test_dataset)
print("\nScratch model test accuracy:", metrics_scratch["eval_accuracy"])

# -----------------------------
# Comparison plot
# -----------------------------
print("\nGenerating comparison plot...")

plt.figure(figsize=(8, 6))
plt.bar(
    ["Pretrained (fine-tuned)", "From Scratch"],
    [metrics_pretrained["eval_accuracy"], metrics_scratch["eval_accuracy"]],
    color=["green", "gray"]
)
plt.title("CIFAR-10: ViT Fine-tuning vs Training from Scratch")
plt.ylabel("Test Accuracy")
plt.ylim(0, 1.0)
for i, v in enumerate([metrics_pretrained["eval_accuracy"], metrics_scratch["eval_accuracy"]]):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center', va='bottom')
plt.tight_layout()
plt.savefig("comparison_vit_full.png")
plt.close()

# -----------------------------
# (f) Attention visualization
# -----------------------------
def visualize_attention(model, processor, original_dataset, save_path="attention_overlay.png"):
    """Visualize attention map overlay on a sample image."""
    print("Generating attention visualization...")
    model.eval()

    # Get a sample from original dataset
    sample = original_dataset[0]
    img = sample['img']

    # Process the image
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Get attention from last layer, average over heads
    attn = outputs.attentions[-1][0].mean(0)[0, 1:].reshape(14, 14).numpy()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(img)
    im = ax[1].imshow(attn, cmap="jet", alpha=0.6, extent=(0, img.size[0], img.size[1], 0))
    ax[1].set_title("Attention Heatmap")
    ax[1].axis("off")

    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Attention map saved to {save_path}")

try:
    visualize_attention(model_pretrained, processor, test_ds)
except Exception as e:
    print(f"Attention visualization failed: {e}")

# -----------------------------
# (g) Summary printout
# -----------------------------
print("\n========== SUMMARY ==========")
print(f"Fine-tuned ViT accuracy: {metrics_pretrained['eval_accuracy']:.4f}")
print(f"Scratch ViT accuracy:    {metrics_scratch['eval_accuracy']:.4f}")
print("=============================")
print("All results and plots saved in current directory.")