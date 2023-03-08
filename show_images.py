import torch
import torchvision
import matplotlib.pyplot as plt

from stanford_cars.dataset import car_dataloader, tensor_to_img


def show_images(dataset, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15, 15))
    for i, tensor in enumerate(dataset):
        tensor = tensor[0]
        img = tensor_to_img(tensor)
        if i == num_samples:
            break
        plt.subplot(num_samples // cols + 1, cols, i + 1)
        plt.imshow(img)


if __name__ == "__main__":
    show_images(car_dataloader)
