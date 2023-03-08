import os
import numpy as np
import torch

from torch import Tensor
from typing import cast
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, image
from torchvision import transforms as T
from PIL import Image
from torchvision import transforms
from numpy import ndarray

torch_img_transform = transforms.Compose([
    # numpy array will have channels permuted to the second index: (b, c, h, w)
    transforms.ToTensor(),
    # normalize from [0, 1] to [-1, 1]
    transforms.Lambda(lambda x: x * 2 - 1)
])


def tensor_to_img(tensor):
    tensor = tensor[0]
    tensor = cast(Tensor, 255 * (tensor + 1) / 2)
    img = cast(ndarray, tensor.permute(1, 2, 0).cpu().numpy()).astype("uint8")
    img = Image.fromarray(img)
    return img


class CarDataset(Dataset):
    def __init__(self):
        super(CarDataset, self).__init__()
        self.data = []
        self.load_data()

    def load_data(self):
        if len(self.data) == 0:
            parent_dir = os.path.dirname(__file__)
            data_dir = os.path.sep.join([parent_dir, "cars_train"])
            for img_name in os.listdir(data_dir):
                img_full_path = os.path.sep.join([data_dir, img_name])
                self.data.append(img_full_path)
            print("data loaded")

    def __getitem__(self, index):
        img = Image.open(self.data[index])
        img = img.resize((64, 64), Image.BILINEAR)
        tensor = torch_img_transform(img)
        channel_dim = tensor.shape[0]
        if channel_dim != 3:
            print("image {} does not have 3 channels".format(self.data[index]))
        return tensor

    def __len__(self):
        return len(self.data)


car_dataloader = DataLoader(dataset=CarDataset(),
                            batch_size=1,
                            shuffle=True,
                            drop_last=True)


if __name__ == "__main__":
    dataset = CarDataset()
    pil_img = tensor_to_img(dataset[10])
    pil_img.save("test.png")
