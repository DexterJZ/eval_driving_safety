import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from Model import CNN
from Dataset import DynamicVehicleDataset
import os
from tqdm import tqdm


device = ("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.3091, 0.3181, 0.3248),
                                 (0.2328, 0.2308, 0.2337)),
        ]
    )

num_epochs = 20
lr = 0.000001
train_CNN = False
batch_size = 32
shuffle = True
pin_memory = True
num_workers = 1

training_set = DynamicVehicleDataset('data/training_image_2/',
                                     'training_csv.csv',
                                     transform)

validation_set = DynamicVehicleDataset('data/validation_image_2/',
                                       'validation_csv.csv',
                                       transform)

training_loader = DataLoader(dataset=training_set, shuffle=shuffle,
                             batch_size=batch_size, num_workers=num_workers,
                             pin_memory=pin_memory)

validation_loader = DataLoader(dataset=validation_set, shuffle=shuffle,
                               batch_size=batch_size, num_workers=num_workers,
                               pin_memory=pin_memory)

model = CNN().to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for name, param in model.vgg16.named_parameters():
    if 'classifier' in name:
        param.requires_grad = True
    else:
        param.requires_grad = train_CNN


def check_accuracy(loader, model):
    if loader == training_loader:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on validation data")

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

    print(
            f"Got {num_correct} / {num_samples} with accuracy {accuracy}"
        )

    model.train()


def train():
    model.train()

    for epoch in range(1, num_epochs + 1):
        loop = tqdm(training_loader, total=len(training_loader), leave=True)

        for imgs, labels in loop:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

        if epoch % 2 == 0:
            loop.set_postfix(val_acc=check_accuracy(validation_loader, model))

        if epoch >= 2 and optimizer.param_groups[0]['lr'] == 0.000001:
            optimizer.param_groups[0]['lr'] *= 0.5

        if epoch >= 4 and \
                optimizer.param_groups[0]['lr'] == 0.00000005:
            optimizer.param_groups[0]['lr'] *= 0.5

        save_name = os.path.join('model/', 'cnn_{}.pth'.format(epoch))

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, save_name)


if __name__ == "__main__":
    train()
