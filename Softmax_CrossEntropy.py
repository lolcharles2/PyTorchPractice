import torch
import torch.nn as nn

# Softmax
x = torch.tensor([2.0, 1.0, 0.1])
outs = torch.softmax(x, dim=0)
print(outs)


# for cross entropy loss, actual y must contain CLASS LABELS, not one hot
# predicted y must contain logit scores, not softmax values

# for BINARY classification with sigmoid, must implement sigmoid manually and use BCE loss

loss = nn.CrossEntropyLoss()
y = torch.tensor([2, 0, 1])
y_pred_good = torch.tensor([[2.0, 1.0, 6.1],
                            [3.0, 2.0, 1.1],
                            [1.0, 8.0, 6.1]])
y_pred_bad = torch.tensor([[0.5, 2.0, 0.2],
                           [1.0, 1.0, 2.1],
                           [1.0, 2.0, 0.6]])

l1 = loss(y_pred_good, y)
l2 = loss(y_pred_bad, y)
print(l1.item())
print(l2.item())

_, predictions1 = torch.max(y_pred_good, 1)
_, predictions2 = torch.max(y_pred_bad, 1)
print(predictions1)
print(predictions2)