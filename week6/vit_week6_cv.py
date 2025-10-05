# vit_cifar10.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import ViTForImageClassification, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import json

# --- Preprocessing ---
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# --- Load CIFAR-10 ---
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_size = int(0.9 * len(trainset))
val_size = len(trainset) - train_size
train_ds, val_ds = torch.utils.data.random_split(trainset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(testset, batch_size=32, shuffle=False)

# --- Model ---
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=10,
    ignore_mismatched_sizes=True
)

# Freeze backbone, only train classifier
for param in model.vit.parameters():
    param.requires_grad = False

# --- Optimizer & Scheduler ---
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 10
num_training_steps = num_epochs * len(train_loader)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=num_training_steps
)

loss_fn = torch.nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_accs, val_accs, train_losses, val_losses = [], [], [], []

# --- Training Loop ---
for epoch in range(num_epochs):
    model.train()
    train_loss, correct, total = 0, 0, 0
    for batch in train_loader:
        optimizer.zero_grad()
        inputs, labels = batch[0].to(device), batch[1].to(device)
        outputs = model(inputs).logits
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_acc = 100 * correct / total

    # Validation
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs).logits
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = 100 * correct / total

    train_accs.append(train_acc)
    val_accs.append(val_acc)
    train_losses.append(train_loss / len(train_loader))
    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}%, Val Acc {val_acc:.2f}%")

# --- Test Evaluation ---
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        outputs = model(inputs).logits
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
test_acc = 100 * correct / total
print(f"Test Accuracy: {test_acc:.2f}%")

# --- Save Results ---
torch.save(model.state_dict(), "vit_cifar10.pth")

metrics = {
    "train_accs": train_accs,
    "val_accs": val_accs,
    "train_losses": train_losses,
    "val_losses": val_losses,
    "test_acc": test_acc
}
with open("results.json", "w") as f:
    json.dump(metrics, f)

plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.legend()
plt.savefig("accuracy_curve.png")

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.savefig("loss_curve.png")
