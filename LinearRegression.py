import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Generating some data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise = 20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)

n_samples, n_features = X.shape

# Define model
model = nn.Linear(n_features, 1)

# Loss and optimizer
criterion = nn.MSELoss()
op = torch.optim.SGD(model.parameters(), lr=0.01)

# Training loop
n_epochs = 1000
for epoch in range(n_epochs):
    y_pred = model(X)

    loss = criterion(y_pred, y)

    # Compute gradients
    loss.backward()

    # Update weights
    op.step()

    # Clears grad computations
    op.zero_grad()

    if epoch % 10 == 0:
        print(f'epoch: {epoch+1}, loss: {loss.item():.4f}')

plt.figure()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, model(X).detach().numpy())
plt.show()
