import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# import dataset from torch
import torch
from torch.utils.data import Dataset
from typing import Callable
import cv2

class PneumoDataset(Dataset):

    def __init__(
        self,
        root,
        images_dir,
        masks_dir,
        csv,
        id_col="DICOM",
        aug_col="Augmentation",
        transform:callable = None,
        augment:callable = None,
        mode:list[str] = ["image", "mask"]
    ):
        images_dir = os.path.join(root, images_dir)
        masks_dir = os.path.join(root, masks_dir)
        df = pd.read_csv(os.path.join(root, csv))
        #df = df[df["Pneumothorax"] == 1]

        self.ids = [(r[id_col], r[aug_col]) for i, r in df.iterrows()]
        self.images = [os.path.join(images_dir, item[0] + ".png") for item in self.ids]
        self.masks = [
            os.path.join(masks_dir, item[0] + "_mask.png") for item in self.ids
        ]
        self.mode = mode
        self.transform = transform
        self.augment = augment

    def __getitem__(self, i):

        image = cv2.imread(self.images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(self.masks[i], 0) == 255).astype("float")
        mask = np.expand_dims(mask, axis=-1)

        #image = image.astype(np.float32)

        if self.transform:
            image, mask = self.transform(image, mask, self.mode)
        
        if self.augment:
            image, mask = self.augment(image, mask)

        return image, mask

    def __len__(self):
        return len(self.ids)
