import torch
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)

# --- 1. Load CIFAR-10 ---
train_ds, test_ds = load_dataset("cifar10", split=["train", "test"])
splits = train_ds.train_test_split(test_size=0.1)
train_ds, val_ds = splits["train"], splits["test"]

# --- 2. Preprocessing ---
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")



def transform(example):
    inputs = processor(images=example["image"], return_tensors="pt")
    inputs["labels"] = example["label"]
    return inputs

train_ds.set_transform(transform)
val_ds.set_transform(transform)
test_ds.set_transform(transform)

# --- 3. Model ---
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=10,
    ignore_mismatched_sizes=True
)


# Freezee
# Freezing means setting requires_grad = False on certain layers, which:
#
# Prevents gradient computation for those parameters during backpropagation
#
# No weight updates occur for frozen layers during training
#
# Only the classifier head gets trained (the final layer that maps ViT features to 10 CIFAR-10 classes)
# print("Freezing backbone layers...")
# for name, param in model.named_parameters():
#     if "classifier" not in name:  # Only keep classifier trainable
#         param.requires_grad = False

# --- 4. Metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# --- 5. Training Args ---
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2
)

# --- 6. Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=processor,
    compute_metrics=compute_metrics,
)

# --- 7. Train ---
train_results = trainer.train()

# --- 8. Evaluate ---
metrics = trainer.evaluate(test_ds)
print("Final Test Results:", metrics)

# --- 9. Save Model & Metrics ---
trainer.save_model("./vit_cifar10_model")
processor.save_pretrained("./vit_cifar10_model")

with open("results.json", "w") as f:
    json.dump(metrics, f)

# --- 10. Plot training curves ---
logs = trainer.state.log_history
val_accs = [x["eval_accuracy"] for x in logs if "eval_accuracy" in x]
train_losses = [x["loss"] for x in logs if "loss" in x]

plt.plot(val_accs, label="Validation Accuracy")
plt.legend()
plt.savefig("val_accuracy_curve.png")

plt.plot(train_losses, label="Training Loss")
plt.legend()
plt.savefig("train_loss_curve.png")

# --- 11. Attention Visualization (Optional Task f) ---
def visualize_attention(model, processor, dataset, save_path="attention_maps.png"):
    model.eval()
    batch = next(iter(dataset))  # take first batch
    img = batch["images"]

    # Process image
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions  # tuple of layers
    last_layer_attention = attentions[-1][0]  # take last layer, first head
    cls_attn = last_layer_attention[0, 0, 1:].reshape(14, 14)  # remove CLS, reshape

    # Plot
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(img)
    ax[1].imshow(cls_attn.numpy(), cmap="jet", alpha=0.6, extent=(0, img.size[0], img.size[1], 0))
    ax[1].set_title("Attention Heatmap")
    ax[1].axis("off")

    plt.savefig(save_path)
    print(f"Attention visualization saved to {save_path}")

# Call it on one test image
visualize_attention(model, processor, test_ds, save_path="attention_example.png")
