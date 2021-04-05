import torch.nn as nn
import torchvision.models as models


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ReLU(nn.Module):
    def forward(self, input):
        return nn.functional.relu(input)


class CNN(nn.Module):
    def __init__(self, train_CNN=False, num_classes=1):
        super(CNN, self).__init__()
        self.train_CNN = train_CNN
        self.resnet50 = models.resnet50(pretrained=True)

        classifier = nn.Sequential(
            nn.Linear(2048, num_classes, bias=True),
            nn.Dropout(0.5),
            nn.Sigmoid()
        )

        self.resnet50.fc = classifier

    def forward(self, images):
        return self.resnet50(images).squeeze(1)
