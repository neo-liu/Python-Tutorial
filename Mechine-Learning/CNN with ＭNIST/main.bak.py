# 对main.py进行了改进
# 在每次epoch中对h, c都进行了修正,使用最后一次的h, c进行验证测试
# 最终得到的结果并不理想
# 想知道是什么原因导致这样的事情
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# device configuration.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define ultra-parameters.
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.003

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataset =  torchvision.datasets.MNIST(root='./data',
                                           train=False,
                                           transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x, h0, c0):
        # set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        # forward propagate LSTM
        out, states = self.lstm(x, (h0, c0))

        # decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out, states


model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)
print(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train the model
h, c = None, None
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        # forward pass
        outputs, states = model(images, h, c)
        h, c = states
        loss = criterion(outputs, labels)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch: [{}/{}], step: [{}/{}], loss: {:.5f}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# test the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs, _ = model(images, h, c)
        _, prediction = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (prediction == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
