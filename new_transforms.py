import albumentations as A
import random
import cv2
import numpy as np
import torch


def augment(image: np.ndarray, mask: np.ndarray | None = None, aug_key:int|None=None, resize_dim=512):

    resize = A.Resize(width=resize_dim, height=resize_dim, interpolation=cv2.INTER_AREA)
    normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1)

    augmentations = {
        0: A.Compose([]),
        1: A.Compose([A.HorizontalFlip(p=1), A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1)]),
        2: A.Compose([A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1)]),
        3: A.Compose([A.RandomGamma(gamma_limit=(60, 120), p=1)]),
        4: A.Compose([A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1)]),
        5: A.Compose([A.OpticalDistortion(distort_limit=0.2, p=1)]),
        6: A.Compose([
            A.Affine(scale=(0.95, 1.05), rotate=(-15, 15), translate_percent=(0.05, 0.05), p=1),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1)
        ]),
        7: A.Compose([
            A.HorizontalFlip(p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.RandomGamma(gamma_limit=(60, 120), p=1),
        ]),
        8: A.Compose([
            A.HorizontalFlip(p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1),
        ]),
        9: A.Compose([
            A.HorizontalFlip(p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.OpticalDistortion(distort_limit=0.2, p=1),
        ]),
        10: A.Compose([
            A.MotionBlur(blur_limit=(3, 5), p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1)
        ]),
        11: A.Compose([
            A.Affine(scale=(0.95, 1.05), rotate=(-15, 15), translate_percent=(0.05, 0.05), p=1),
            A.GridDistortion(p=1)
        ]),
        12: A.Compose([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1),
            A.MedianBlur(blur_limit=(3, 5), p=1)
        ]),
        13: A.Compose([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1),
            A.OpticalDistortion(distort_limit=0.2, p=1)
        ]),
        14: A.Compose([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1),
            A.RandomGamma(gamma_limit=(60, 120), p=1)
        ]),
        15: A.Compose([
            A.RandomGamma(gamma_limit=(60, 120), p=1),
            A.OpticalDistortion(distort_limit=0.2, p=1),
        ]),
        16: A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1),
        ]),
        17: A.Compose([
            A.Affine(scale=(0.95, 1.05), rotate=(-15, 15), translate_percent=(0.05, 0.05), p=1),
            A.RandomGamma(gamma_limit=(60, 120), p=1),
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=1)
        ]),
        18: A.Compose([
            A.Affine(scale=(0.95, 1.05), rotate=(-15, 15), translate_percent=(0.05, 0.05), p=1),
            A.HorizontalFlip(p=1),
            A.GridDistortion(p=1),
        ]),
        19: A.Compose([
            A.Affine(scale=(0.95, 1.05), rotate=(-15, 15), translate_percent=(0.05, 0.05), p=1),
            A.OpticalDistortion(distort_limit=0.2, p=1),
        ]),
    }

    random_key = random.choice(list(augmentations.keys())) if aug_key is None else aug_key
    aug = augmentations[random_key]
    
    if mask is not None:
        resized = resize(image=image, mask=mask)
        augmented = aug(image=resized['image'], mask=resized['mask'])
        normalized = normalize(image=augmented['image'], mask=augmented['mask'])
        return normalized['image'], normalized['mask']
    else:
        resized = resize(image=image)
        augmented = aug(image=resized['image'])
        normalized = normalize(image=augmented['image'])
        return normalized['image'], None
    


def complete_transform(image, mask, aug_key=None):
    image, mask = augment(image, mask, aug_key)
    image = torch.permute(2, 0, 1).unsqueeze(0)
    mask = torch.permute(2, 0, 1).unsqueeze(0)
    return image, mask


    
