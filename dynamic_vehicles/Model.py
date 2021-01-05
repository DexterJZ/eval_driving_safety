import torch.nn as nn
import torchvision.models as models


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class CNN(nn.Module):
    def __init__(self, train_CNN=False, num_classes=1):
        super(CNN, self).__init__()
        self.train_CNN = train_CNN
        self.vgg16 = models.vgg16(pretrained=True)

        classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(0.5),
            nn.Linear(self.vgg16.classifier[0].in_features, num_classes),
            nn.Sigmoid()
        )

        self.vgg16.classifier = classifier

    def forward(self, images):
        return self.vgg16(images).squeeze(1)
