import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
# Generate data
np.random.seed(0)
n = 5000
x = np.random.uniform(-10, 10, (n, 2))
y = x[:, 0]**2 + x[:, 0]*x[:, 1] + x[:, 1]**2
x_train, y_train = x[:int(n*0.9)], y[:int(n*0.9)]
x_test, y_test = x[int(n*0.9):], y[int(n*0.9):]
# Define model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
# Train model
train_losses = []
test_losses = []
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = net(torch.Tensor(x_train))
    loss = criterion(outputs, torch.Tensor(y_train).unsqueeze(1))
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    with torch.no_grad():
        test_outputs = net(torch.Tensor(x_test))
        test_loss = criterion(test_outputs,
        torch.Tensor(y_test).unsqueeze(1))
        test_losses.append(test_loss.item())
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Train Loss: {loss.item()}, Test Loss: {test_loss.item()}")
# Plot losses
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.legend()
plt.show()
# Test model
with torch.no_grad():
    test_outputs = net(torch.Tensor(x_test))
    test_loss = criterion(test_outputs, torch.Tensor(y_test).unsqueeze(1))
    print(f"Test Loss: {test_loss.item()}")
