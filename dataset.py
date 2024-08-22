import cv2
import torch
import random
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from skimage import data
from torch.utils.data import Dataset

class AugmentedDataset(Dataset):

    def __init__(self, images_tensor, masks_tensor, augment):
        self.images = images_tensor
        self.masks = masks_tensor
        self.augment = augment
        self.augmentations = [self.flip, self.dropout, self.contrast, self.brightness, self.rotate]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        if self.augment and random.random() > 0.1:
            image, mask = self.augment_image_and_mask(image, mask)
        return image, mask

    def augment_image_and_mask(self, image, mask):
        image = image.clone()
        mask = mask.clone()
        for function in self.augmentations:
            if random.random() > 0.5:
                image, mask = function(image, mask)
        return image, mask

    def flip(self, image, mask):
        axis = random.choice([[1], [2], [1, 2]])
        image = torch.flip(image, axis)
        mask = torch.flip(mask, axis)
        return image, mask

    def dropout(self, image, mask):
        dropout_mask = torch.rand_like(image) > 0.001
        image = image * dropout_mask
        return image, mask

    def contrast(self, image, mask):
        mean = torch.mean(image, dim=(1, 2), keepdim=True)
        image = (image - mean) * random.uniform(0.5, 1.5) + mean
        return torch.clamp(image, 0, 1), mask

    def brightness(self, image, mask):
        image = image * random.uniform(0.5, 1.5)
        return torch.clamp(image, 0, 1), mask

    def rotate(self, image, mask):
        angle = random.choice([90, 180, 270])
        image = F.rotate(image, angle)
        mask = F.rotate(mask, angle)
        return image, mask

    def visualise_augmentation(self, function):    
        image, mask = self.images, self.masks
        image_aug, mask_aug = function(image, mask)
        fig, axs = plt.subplots(2, 2, figsize=(4, 4))
        axs[0, 0].imshow(image.permute(1, 2, 0).cpu().numpy())
        axs[0, 1].imshow(image_aug.permute(1, 2, 0).cpu().numpy())
        axs[1, 0].imshow(mask.squeeze().cpu().numpy(), cmap='gray')
        axs[1, 1].imshow(mask_aug.squeeze().cpu().numpy(), cmap='gray')
        plt.show()

if __name__ == "__main__":
    image = cv2.resize(data.cat(), (128, 128))
    mask = (image[:, :, 0] > 128).astype(float)
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)    
    dataset = AugmentedDataset(image_tensor, mask_tensor, augment=True)
    dataset.visualise_augmentation(dataset.rotate)