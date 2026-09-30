import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = "./data"
MEAN = 0.1307
STD = 0.3081
VALID_RATIO = 0.9
BATCH_SIZE = 256
NUM_EPOCHS = 10

train_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[MEAN], std=[STD])]
)
test_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[MEAN], std=[STD])]
)

full_train_data = datasets.MNIST(
    root=ROOT, train=True, download=True, transform=train_transforms
)
test_data = datasets.MNIST(root=ROOT, train=False, download=True, transform=test_transforms)

n_train_examples = int(len(full_train_data) * VALID_RATIO)
n_valid_examples = len(full_train_data) - n_train_examples
train_data, valid_data = data.random_split(
    full_train_data, [n_train_examples, n_valid_examples]
)

train_dataloader = data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
valid_dataloader = data.DataLoader(valid_data, batch_size=BATCH_SIZE)
test_dataloader = data.DataLoader(test_data, batch_size=BATCH_SIZE)


class LeNetClassifier(nn.Module):
    """LeNet-style convolutional classifier for 28x28 grayscale MNIST images."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, inputs):
        outputs = F.relu(self.avgpool1(self.conv1(inputs)))
        outputs = F.relu(self.avgpool2(self.conv2(outputs)))
        outputs = self.flatten(outputs)
        outputs = F.relu(self.fc_1(outputs))
        outputs = F.relu(self.fc_2(outputs))
        return self.fc_3(outputs)


def train(model, optimizer, criterion, dataloader, device, epoch=0, log_interval=50):
    model.train()
    epoch_acc, epoch_count = 0, 0
    running_acc, running_count = 0, 0
    losses = []
    start_time = time.time()

    for idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, labels)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        correct = (predictions.argmax(1) == labels).sum().item()
        batch_size = labels.size(0)
        epoch_acc += correct
        epoch_count += batch_size
        running_acc += correct
        running_count += batch_size

        if idx % log_interval == 0 and idx > 0:
            print(
                f"| epoch {epoch:3d} | {idx:5d}/{len(dataloader):5d} batches | "
                f"accuracy {running_acc / running_count:8.3f}"
            )
            running_acc, running_count = 0, 0
            start_time = time.time()

    return epoch_acc / epoch_count, sum(losses) / len(losses)


def evaluate(model, criterion, dataloader, device):
    model.eval()
    total_acc, total_count = 0, 0
    losses = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            predictions = model(inputs)
            loss = criterion(predictions, labels)
            losses.append(loss.item())
            total_acc += (predictions.argmax(1) == labels).sum().item()
            total_count += labels.size(0)

    return total_acc / total_count, sum(losses) / len(losses)


num_classes = len(train_data.dataset.classes)
lenet_model = LeNetClassifier(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lenet_model.parameters())

save_model = "./model"
os.makedirs(save_model, exist_ok=True)
model_path = os.path.join(save_model, "lenet_model.pt")

best_loss_eval = float("inf")
for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    train_acc, train_loss = train(
        lenet_model, optimizer, criterion, train_dataloader, device, epoch
    )
    eval_acc, eval_loss = evaluate(lenet_model, criterion, valid_dataloader, device)

    if eval_loss < best_loss_eval:
        best_loss_eval = eval_loss
        torch.save(lenet_model.state_dict(), model_path)
        print(f"--> Saved best checkpoint at epoch {epoch} (validation loss: {eval_loss:.4f})")

    print(
        f"Epoch {epoch:2d} | Train Acc: {train_acc:.4f} | "
        f"Val Acc: {eval_acc:.4f} | Time: {time.time() - epoch_start_time:.2f}s"
    )
    print("-" * 59)

test_acc, test_loss = evaluate(lenet_model, criterion, test_dataloader, device)
print(f"Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")
