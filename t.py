from unittest import TestResult
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from tqdm import tqdm

classes = ("animal", "person")
PATH = './cifar_net.pth'
batch_size = 32

class conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel, padding=1):
      super().__init__()
      self.conv1 = nn.Conv2d(in_c, out_c, kernel, padding=padding)
      self.conv1_1 = nn.Conv2d(out_c, out_c, kernel, padding=padding)
      self.bn_conv1 = nn.BatchNorm2d(out_c)
      self.bn_conv1_1 = nn.BatchNorm2d(out_c)
      self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
      x = F.relu(self.bn_conv1(self.conv1(x)))
      x = self.pool(F.relu(self.conv1_1(x)))
      return x

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    # self.conv1 = conv_block(3,32,3,1)
    # self.conv2 = conv_block(32,64,3,1)
    # self.conv3 = conv_block(64,128,3,1)
    # self.conv4 = conv_block(128,256,3,1)
    
    self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
    self.conv1_1 = nn.Conv2d(64, 64, 3, padding=1)
    self.bn_conv1 = nn.BatchNorm2d(64)
    self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
    self.bn_conv2 = nn.BatchNorm2d(128)
    self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
    self.bn_conv3 = nn.BatchNorm2d(256)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(256 * 14 * 14, 128)
    self.fc2 = nn.Linear(128, 84)
    self.fc3 = nn.Linear(84, 2)
    self.dropout = nn.Dropout(.3)

  def forward(self, x):
    # x = self.conv1(x)
    # x = self.conv2(x)
    # x = self.conv3(x)
    #x = self.conv4(x)

    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.bn_conv1(self.conv2(x))))
    x = self.pool(F.relu(self.conv3(x)))
    x = self.pool(F.relu(self.conv4(x)))
    x = torch.flatten(x, 1)  # flatten all dimensions except batch
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.dropout(self.fc3(x))
    x = F.log_softmax(x, dim=1)
    return x

net = Net()

def imshow(img):
  img = img / 2 + 0.5  # unnormalize
  npimg = img.numpy()
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

def check_accuarcy(testloader, epoch):
  correct_pred = {classname: 0 for classname in classes}
  total_pred = {classname: 0 for classname in classes}

  # again no gradients needed
  with torch.no_grad():
    for data in testloader:
      images, labels = data
      outputs = net(images)
      _, predictions = torch.max(outputs, 1)
      # collect the correct predictions for each class
      for label, prediction in zip(labels, predictions):
        if label == prediction:
          correct_pred[classes[label]] += 1
        total_pred[classes[label]] += 1

  # print accuracy for each class
  print('---- Epoch {} ----'.format(epoch))
  for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

def train():
  train_transform = transforms.Compose(
    [
      transforms.Resize((224, 224)),
      # transforms.RandomHorizontalFlip(.1),
      # transforms.RandomSolarize(.1),
      # transforms.RandomEqualize(.1),
      # transforms.RandomGrayscale(.1),
      transforms.ToTensor()])
  test_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor()])

  trainset = datasets.ImageFolder("input/trainanimals/train",
                  transform=train_transform)

  testset = datasets.ImageFolder("input/trainanimals/test",
                   transform=test_transform)

  trainloader = torch.utils.data.DataLoader(trainset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=1)

  testloader = torch.utils.data.DataLoader(testset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=1)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(net.parameters(), lr=0.001)
  # prepare to count predictions for each class

  for epoch in range(5):  # loop over the dataset multiple times
    running_loss = 0.0
    counter = 0
    loop = tqdm(trainloader, ncols=50)
    for i, data in enumerate(loop):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = net(inputs)
      #loss = criterion(outputs, labels)
      loss = F.nll_loss(outputs, labels)
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()
      counter += labels.shape[0]
      loop.set_postfix(loss=(running_loss / (i+1)))
    #print(f"[{epoch + 1}] loss: {running_loss / counter:.3f}")
    check_accuarcy(testloader=testloader, epoch=epoch+1)
    

  print("Finished Training")

  torch.save(net.state_dict(), PATH)

def test():
  net = Net()
  net.load_state_dict(torch.load(PATH))

  transform = transforms.Compose(
    [transforms.Resize((256, 256)),
     transforms.ToTensor()])

  testset = datasets.ImageFolder("input/trainanimals/test",
                   transform=transform)

  testloader = torch.utils.data.DataLoader(testset,
                        batch_size=2,
                        shuffle=True,
                        num_workers=2)

  # prepare to count predictions for each class
  correct_pred = {classname: 0 for classname in classes}
  total_pred = {classname: 0 for classname in classes}

  # again no gradients needed
  with torch.no_grad():
    for data in testloader:
      images, labels = data
      outputs = net(images)
      _, predictions = torch.max(outputs, 1)
      # collect the correct predictions for each class
      for label, prediction in zip(labels, predictions):
        result = "Failed"
        if label == prediction:
          correct_pred[classes[label]] += 1
          result = "Success"
        total_pred[classes[label]] += 1
        #print("prediction: {} actual: {} => {}".format(prediction, label, result))

  for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

if __name__ == "__main__":
  if sys.argv[1] == 'train':
    train()
  elif sys.argv[1] == 'test':
    test()
