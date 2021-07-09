import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Preparing some data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Scaling features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Converting to torch tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# Setting up model
# f = sigmoid(wx + b)

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegression(n_features)

# Loss function
learning_rate = 0.01
criterion = nn.BCELoss()
op = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
epochs = 300

for epoch in range(epochs):
    # Computing predictions
    y_pred = model(X_train)

    # Computing loss
    loss = criterion(y_pred, y_train)

    # Computing gradients
    loss.backward()

    # Updating weights
    op.step()

    op.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch + 1}, loss: {loss.item():.4f}')

with torch.no_grad():
    # Computing test accuracy

    y_pred = model(X_test)
    class_labels = y_pred > 0.5

    accuracy = (class_labels == y_test).double().mean()

    print(f'Test accuracy: {accuracy:.4f}')