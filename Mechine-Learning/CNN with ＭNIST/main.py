# import tools.
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# fixed operation to choose which device to use.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define ultra-parameters.
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# set training and testing data set with some setting.
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=False)
test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# define convolutional class and processions inside of convolutional class.
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        # redefine convolutional class combined with ultra-parameters.
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # construct fully-connected nerual network as usual to convert 2D signal into 1D signal.
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        out = out.reshape(out.size(0), -1)
        return self.fc(out)


model = ConvNet(num_classes).to(device)
print(model)

# define optimizer and loss function.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training.
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)

        loss = criterion(predictions, labels)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch: [{}/{}], Step: [{}/{}], loss: {}'
                  .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# validation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, Predictions = torch.max(outputs, dim=1)
        correct += (Predictions == labels).sum().item()
        total += labels.size(0)

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

torch.save(model.state_dict(), 'cnn_with_mnist.ckpt')
