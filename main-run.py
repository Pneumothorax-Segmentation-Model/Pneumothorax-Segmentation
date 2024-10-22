# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
from segmentation_models_pytorch.utils.base import Loss
from segmentation_models_pytorch.base.modules import Activation
import wandb

# %%
DEVICE = "cuda" if torch.cuda.is_available() else "mps"
print(DEVICE)
torch.cuda.empty_cache()
RESUME = 0
NUM_EPOCHS = 10
#ENCODER = "resnet50"
##ENCODER = "efficientnet-b7"
ENCODER = "se_resnext50_32x4d"

BATCH_16 =16
BATCH_8 = 8

DEST_DIR = f"./{ENCODER}_BS_Focal_Aggressive"
if not os.path.exists(DEST_DIR):
    os.makedirs(DEST_DIR)

# %%
wandb.init(
    # Set the project where this run will be logged
    project="SE-ResNeXt50-BS_Focal_Aggressive",
    # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
    name=f"experiment_focal_agg-aug-seresnext50_BS_lr_plateau_combo_loss_0_39",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": 0.0001,
        "architecture": ENCODER,
        "dataset": "SIIM-ACR",
        "epochs": NUM_EPOCHS,
    },
)

# %%
# root = "../../Data"
root = "./Data"
print(os.getcwd())
print(os.listdir(root))


images_dir = "bone_suppressed_files"
#images_dir = "png_files"
masks_dir = "mask_files"
train_csv = "csv/train_upsampled.csv"
val_csv = "csv/val_final.csv"

# %%
if not os.path.exists(DEST_DIR+"/models"):
    os.mkdir(DEST_DIR+"/models")

if not os.path.exists(DEST_DIR+"/logs"):
    os.mkdir(DEST_DIR+"/logs")

# %%
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()

# %%
class Dataset(BaseDataset):

    def __init__(
        self,
        root,
        images_dir,
        masks_dir,
        csv,
        aug_fn=None,
        id_col="DICOM",
        aug_col="Augmentation",
        preprocessing_fn=None,
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
        self.aug_fn = aug_fn
        self.preprocessing_fn = preprocessing_fn

    def __getitem__(self, i):

        image = cv2.imread(self.images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(self.masks[i], 0) == 255).astype("float")
        mask = np.expand_dims(mask, axis=-1)

        #image = image.astype(np.float32)

        aug = self.ids[i][1]
        # if aug:
        augmented = self.aug_fn(aug)(image=image, mask=mask)
        image, mask = augmented["image"], augmented["mask"]

        if self.preprocessing_fn:
            sample = self.preprocessing_fn(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.ids)


# %%

from albumentations import (
    HorizontalFlip,
    RandomBrightnessContrast,
    RandomGamma,
    CLAHE,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    ShiftScaleRotate,
    Normalize,
    GaussNoise,
    Compose,
    Lambda,
    Resize,
    MedianBlur,
    MotionBlur
)


def augmentation_fn(value, resize=21):
    augmentation_options = {
        0: [],
        1: [HorizontalFlip(p=1),
            CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1),
        ],
        2: [RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1)],
        3: [RandomGamma(gamma_limit=(60, 120), p=1)],
        4: [CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1)],
        5: [OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)],
        6: [ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=15, p=1),
            ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1),
        ],
        7: [GaussNoise(p=1)],
        8: [
            HorizontalFlip(p=1),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            RandomGamma(gamma_limit=(60, 120), p=1),
        ],
        9: [
            HorizontalFlip(p=1),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1),
        ],
        10: [
            HorizontalFlip(p=1),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1),
        ],
        11: [
            MotionBlur(blur_limit=4, p=1),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            GaussNoise(p=1),
        ],
        12: [
            ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=15, p=1),
            GaussNoise(p=1),
            GridDistortion(p=1)
        ],
        13: [CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1), MedianBlur(blur_limit=4, p=1)],
        14: [CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1), OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)],
        15: [CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1), RandomGamma(gamma_limit=(60, 120), p=1)],
        16: [RandomGamma(gamma_limit=(60, 120), p=1), OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)],
        17: [
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            GaussNoise(p=1),
            CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1),
        ],
        18: [
            ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=15, p=1),
            RandomGamma(gamma_limit=(60, 120), p=1),
            ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1),
        ],
        19: [
            ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=15, p=1),
            HorizontalFlip(p=1),
            GridDistortion(p=1),
        ],
        20: [
            ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=15, p=1),
            GaussNoise(p=1),
            OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1),
        ],
        21: [Resize(width=512, height=512, interpolation=cv2.INTER_AREA)],

        # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    }

    return Compose(augmentation_options[resize] + augmentation_options[value],is_check_shapes=False)



# %%
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        Lambda(image=preprocessing_fn),
        Lambda(image=to_tensor, mask=to_tensor),
    ]
    return Compose(_transform)

# %%
# dataset = Dataset(
#     root=root,
#     images_dir=images_dir,
#     masks_dir=masks_dir,
#     csv=train_csv,
#     aug_fn=augmentation_fn,
# )
# print(len(dataset))
# image, mask = dataset[1]  # get some sample
# print(image.shape)
# print(mask.shape)
# print(np.unique(mask.squeeze()))
# print(set(mask.flatten()))
# visualize(
#     image=image,
#     mask=mask.squeeze(),
# )

# %%

def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        # _log_api_usage_once(sigmoid_focal_loss)
    p = inputs
    ce_loss = torch.nn.functional.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss

class FocalLoss(Loss):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', activation=None, ignore_channels=None, **kwargs):
        """
        Args:
            alpha (float, optional): Weighting factor for positive examples. Default is 0.25.
            gamma (float, optional): Exponent of the modulating factor to focus on hard examples. Default is 2.0.
            reduction (str, optional): Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'. Default is 'mean'.
            activation (str, optional): Activation function to apply to the model outputs (e.g., 'sigmoid').
            ignore_channels (list, optional): List of channels to ignore when computing the loss. Default is None.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return focal_loss(
            y_pr,
            y_gt,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
            ignore_channels=self.ignore_channels,
        )
# class WeightedSumOfLosses(Loss):
#     def __init__(self, l1, l2, w1=1, w2=2):
#         # name = "{} + {}".format(l1.__name__, l2.__name__)
#         name = "combo_loss"
#         super().__init__(name=name)
#         self.l1 = l1
#         self.l2 = l2
#         self.w1 = w1
#         self.w2 = w2

#     def __call__(self, *inputs):
#         return self.w1 * self.l1.forward(*inputs) + self.w2 * self.l2.forward(*inputs)
    
class WeightedSumOfLosses(Loss):
    def __init__(self, l1, l2, l3, w1=1, w2=2, w3=2):
        # name = "{} + {}".format(l1.__name__, l2.__name__)
        # BCE:3, Dice:1, Focal:4
        # BCE:2, Dice:1, Focal:2
        name = "combo_loss"
        super().__init__(name=name)
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def __call__(self, *inputs):
        return self.w1 * self.l1.forward(*inputs) + self.w2 * self.l2.forward(*inputs) + self.w3 * self.l3.forward(*inputs)

# %%
epoch = -1
max_score = 0
val_loss = 0
train_loss = 0
train_iou = 0
val_iou = 0
train_dice = 0
val_dice = 0
batch_size = 0
loss_name = "combo_loss"

ACTIVATION = "sigmoid"
ENCODER_WEIGHTS = "imagenet"
#ENCODER_WEIGHTS = None

model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    activation=ACTIVATION,
).to(DEVICE)


# %%
loss = WeightedSumOfLosses(utils.losses.DiceLoss(), utils.losses.BCELoss(), FocalLoss()) #changed it from BCEWithLogitsLoss
metrics = [utils.metrics.IoU(threshold=0.5), utils.metrics.Fscore()] # added Fscore

optimizer = torch.optim.Adam(
    [
        dict(params=model.parameters(), lr=0.0001),
    ]
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode="min",
    factor=0.6,
    patience=4,
    threshold=0.001,
    threshold_mode="abs",
)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# %%
if RESUME:
    #checkpoint = torch.load(f"{DEST_DIR}/models/model_epoch_9_resnet50.pth")
    #checkpoint = torch.load(f"/home/amog/ml-summer-research/se_resnext101_32x4d/models/model_epoch_18_se_resnext101_32x4d.pth")
    checkpoint = torch.load(f"{DEST_DIR}/models/latest_model_{ENCODER}.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint["epoch"]
    loss_name = checkpoint["loss_name"]
    max_score = checkpoint["max_score"]

    train_loss = checkpoint["train_loss"]
    train_dice = checkpoint["train_dice"]
    train_iou = checkpoint["train_iou"]

    val_loss = checkpoint["val_loss"]
    val_dice = checkpoint["val_dice"]
    val_iou = checkpoint["val_iou"]

    batch_size = checkpoint["batch_size"]
    
    started_lr = checkpoint["started_lr"]


print("Scheduler State Dict Outside: ", scheduler.state_dict())
print("Epoch:", epoch)
print("Loss Function:", loss_name)
print("Max Val Dice Score:", max_score)

print("Batch Size:", batch_size or BATCH_16)

print(f"Train Loss: {train_loss} || Val Loss: {val_loss}")
print(f"Train Dice: {train_dice} || Val Dice: {val_dice}")
print(f"Train IoU: {train_iou} || Val IoU: {val_iou}")
print("Optimizer LR:", optimizer.param_groups[0]["lr"])

# %%
train_dataset = Dataset(
    root=root,
    images_dir=images_dir,
    masks_dir=masks_dir,
    csv=train_csv,
    aug_fn=augmentation_fn,
    preprocessing_fn=get_preprocessing(preprocessing_fn),
)
print(len(train_dataset))

val_dataset = Dataset(
    root=root,
    images_dir=images_dir,
    masks_dir=masks_dir,
    csv=val_csv,
    aug_fn=augmentation_fn,
    preprocessing_fn=get_preprocessing(preprocessing_fn),
)

print(len(val_dataset))

train_loader = DataLoader(train_dataset, batch_size=BATCH_8, shuffle=True, num_workers=2)
valid_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2) # Change to True

# %%
# create epoch runners 
# it is a simple loop of iterating over dataloader`s samples
train_epoch = utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True,
)

valid_epoch = utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True,
)

# %%
train_loss_scores = []
val_loss_scores = []
train_iou_scores = []
val_iou_scores = []
train_dice_scores = []
val_dice_scores = []
starting_lrs = []

for i in range(epoch + 1, epoch + NUM_EPOCHS + 1):

    print("\nEpoch: {}".format(i))
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)

    train_iou_scores.append((i, train_logs["iou_score"]))
    val_iou_scores.append((i, valid_logs["iou_score"]))

    train_loss_scores.append((i, train_logs["combo_loss"]))
    val_loss_scores.append((i, valid_logs["combo_loss"]))

    train_dice_scores.append((i, train_logs["fscore"]))
    val_dice_scores.append((i, valid_logs["fscore"]))

    started_lr = optimizer.param_groups[0]["lr"]
    print("Started with LR:", started_lr)
    scheduler.step(valid_logs["combo_loss"])
    print("Changed LR to:", optimizer.param_groups[0]["lr"])
    starting_lrs.append((i, started_lr))

    if max_score < valid_logs["fscore"]:
        max_score = valid_logs["fscore"]
        print("best valid model as per fscore!")

    checkpoint = {
        "epoch": i,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        
        "loss_name": "combo_loss",
        "max_score": max_score,

        "val_loss": valid_logs["combo_loss"],
        "train_loss": train_logs["combo_loss"],
        "train_iou": train_logs["iou_score"],
        "val_iou": valid_logs["iou_score"],
        "train_dice": train_logs["fscore"],
        "val_dice": valid_logs["fscore"],
        
        "started_lr": started_lr,
        "batch_size": train_loader.batch_size,
    }

    torch.save(checkpoint, f"{DEST_DIR}/models/model_epoch_{i}_{ENCODER}.pth")
    print("model saved!")

    if i == epoch + NUM_EPOCHS:
        torch.save(checkpoint, f"{DEST_DIR}/models/latest_model_{ENCODER}.pth")
        print("latest model saved!")

    # more of these ones here!
    np.savetxt(
        f"{DEST_DIR}/logs/train_iou_scores_{epoch+NUM_EPOCHS}.txt", train_iou_scores, fmt="%.5f"
    )
    np.savetxt(
        f"{DEST_DIR}/logs/val_iou_scores_{epoch+NUM_EPOCHS}.txt", val_iou_scores, fmt="%.5f"
    )
    np.savetxt(
        f"{DEST_DIR}/logs/train_loss_{epoch+NUM_EPOCHS}.txt", train_loss_scores, fmt="%.5f"
    )
    np.savetxt(
        f"{DEST_DIR}/logs/val_loss_{epoch+NUM_EPOCHS}.txt", val_loss_scores, fmt="%.5f"
        )
    
    np.savetxt(
        f"{DEST_DIR}/logs/train_dice_scores_{epoch+NUM_EPOCHS}.txt", train_dice_scores, fmt="%.5f"
    )
    np.savetxt(
        f"{DEST_DIR}/logs/val_dice_scores_{epoch+NUM_EPOCHS}.txt", val_dice_scores, fmt="%.5f"
    )

    np.savetxt(
        f"{DEST_DIR}/logs/started_learning_rates_{epoch+NUM_EPOCHS}.txt", starting_lrs, fmt="%.5f"
    )
    
    logging_wandb = checkpoint.copy()
    del logging_wandb["model_state_dict"]
    del logging_wandb["optimizer_state_dict"]
    del logging_wandb["scheduler_state_dict"]
    
    wandb.log(
        logging_wandb
    )

# %%
wandb.finish()

print("train loss:\n", train_loss_scores)
print("train dice:\n", train_dice_scores)
print("train iou:\n", train_iou_scores)
print()
print("val loss:\n", val_loss_scores)
print("val dice:\n", val_dice_scores)
print("val iou:\n", val_iou_scores)



