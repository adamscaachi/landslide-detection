import torch
import random
from torch.utils.data import Dataset

class AugmentedDataset(Dataset):

    def __init__(self, images_tensor, masks_tensor, augment):
        self.images = images_tensor
        self.masks = masks_tensor
        self.augment = augment
        self.augmentations = [self.flip, self.dropout]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        if self.augment and random.random() > 0.1:
            image, mask = self.augment_image_and_mask(image, mask)
        return image, mask

    def augment_image_and_mask(self, image, mask):
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