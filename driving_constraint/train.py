import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from Model import CNN
from Dataset import DrivingConstraintDataset

# Parameters for dataset and model
shuffle = True
pin_memory = True
num_workers = 1

pretrained = False
num_epochs = 20
lr = 0.001
batch_size = 8

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.3775, 0.3923, 0.3839),
                             (0.3110, 0.3154, 0.3180)),
    ]
)

training_set = DrivingConstraintDataset('data/image_2/',
                                        'training_csv.csv',
                                        transform)
validation_set = DrivingConstraintDataset('data/image_2/',
                                          'validation_csv.csv',
                                          transform)

training_loader = DataLoader(dataset=training_set, shuffle=shuffle,
                             batch_size=batch_size, num_workers=num_workers,
                             pin_memory=pin_memory)
validation_loader = DataLoader(dataset=validation_set, shuffle=shuffle,
                               batch_size=batch_size, num_workers=num_workers,
                               pin_memory=pin_memory)

model = CNN().to(device)

for name, param in model.resnet50.named_parameters():
    if 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = pretrained

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = \
    torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15],
                                         gamma=0.1, last_epoch=-1)


def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            predictions = torch.tensor(
                [1.0 if i >= 0.5 else 0.0 for i in scores]).to(device)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    accuracy = f"{float(num_correct)/float(num_samples)*100:.2f}"

    # return accuracy
    print(f"Got {num_correct} / {num_samples} with accuracy {accuracy}%")

    model.train()


def train():
    model.train()

    for epoch in range(1, num_epochs + 1):
        print("Epoch {}:".format(epoch))
        loop = tqdm(training_loader, total=len(training_loader), leave=True)

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            loop.set_postfix(val_acc=check_accuracy(validation_loader, model))

        if epoch % 5 == 0:
            save_name = os.path.join('model/', 'cnn_{}.pth'.format(epoch))

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, save_name)


if __name__ == "__main__":
    train()
