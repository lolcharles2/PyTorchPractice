import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Digit classification via the MNIST data set

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper params
input_size = 784
hidden_size = 100
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.0005

# MNIST data
train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

examples = iter(train_loader)
samples, labels = examples.next()

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(samples[i][0], cmap='gray')

# Neural net
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
op = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (samples, labels) in enumerate(train_loader):
        # forward
        samples = samples.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        y_pred = model(samples)

        # compute loss
        loss = criterion(y_pred, labels)

        # backward
        loss.backward()

        # Update weights
        op.step()

        # Clear gradients
        op.zero_grad()

        if (i+1) % 10 == 0:
            print(f'epoch: {epoch+1}/{num_epochs}, iteration: {i+1}/{n_total_steps}, loss: {loss.item():.4f}')


# test set
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for samples, labels in test_loader:
        samples = samples.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        output = model(samples)

        _, predictions = torch.max(output, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    print(f'Test accuracy: {100*n_correct/n_samples}%')