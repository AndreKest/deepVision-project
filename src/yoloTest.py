import torch

y_true = torch.tensor([0, 0, 1], dtype=torch.float32)
y_pred  = torch.tensor([0.1, 0.4, 0.7], dtype=torch.float32)

#criterion = torch.nn.BCEWithLogitsLoss()
#loss = criterion(y_pred, y_true)
#print(loss)

criterion = torch.nn.BCEWithLogitsLoss()
x = torch.tensor([[-4.8828, -4.4844, -3.9102, -4.4961, -4.8828], [-5.0938, -4.4102, -3.5195, -5.1289, -5.5039],[-6.6875, -7.2578, -3.0254, -5.8594, -4.5391]])
y_true = torch.tensor([[0.0000, 0.0000, 0.4232, 0.0000, 0.0000],[0.0000, 0.0000, 0.2026, 0.0000, 0.0000],[0.4723, 0.0000, 0.0000, 0.0000, 0.0000]])
loss2 = criterion(x, y_true)
print(loss2)



def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def bce_with_logits_loss(logits, targets):
    sigmoid_logits = sigmoid(logits)
    term1 = targets * torch.log(sigmoid_logits)
    term2 = (1 - targets) * torch.log(1 - sigmoid_logits)
    loss = -(term1 + term2).mean()
    return loss

loss = bce_with_logits_loss(x, y_true)
print(loss)