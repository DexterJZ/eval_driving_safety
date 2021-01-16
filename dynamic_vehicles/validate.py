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

batch_size = 32
shuffle = True
pin_memory = True
num_workers = 1

validation_set = DynamicVehicleDataset('data/validation_image_2/',
                                       'validation_csv.csv',
                                       transform)

validation_loader = DataLoader(dataset=validation_set, shuffle=shuffle,
                               batch_size=batch_size, num_workers=num_workers,
                               pin_memory=pin_memory)

model = CNN().to(device)
model.load_state_dict(
    torch.load("model/pretrained_model/cnn_20.pth")['model_state_dict'])

def validate():
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in validation_loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            predictions = torch.tensor(
                [1.0 if i >= 0.5 else 0.0 for i in scores]).to(device)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    accuracy = f"{float(num_correct)/float(num_samples)*100:.2f}"

    print(f"Got {num_correct} / {num_samples} with accuracy {accuracy}")


if __name__ == "__main__":
    validate()