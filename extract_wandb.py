import wandb
import pandas as pd
import matplotlib.pyplot as plt

api = wandb.Api()
run_0 = api.run("/pneumothorax_segmentation/PneumothoraxSegmentation/runs/ey8sp52r")
run_1_7 = api.run("/pneumothorax_segmentation/PneumothoraxSegmentation/runs/95f3ifns")
run_8 = api.run("/pneumothorax_segmentation/PneumothoraxSegmentation/runs/gvpxrp77")
run_9_11 = api.run("/pneumothorax_segmentation/PneumothoraxSegmentation/runs/1kg577kf")
run_12_21 = api.run("/pneumothorax_segmentation/PneumothoraxSegmentation/runs/2zuv2gxw")
run_22 = api.run("/pneumothorax_segmentation/PneumothoraxSegmentation/runs/16kgy45w")
run_23_32 = api.run("/pneumothorax_segmentation/PneumothoraxSegmentation/runs/1wz8nfmd")
run_33 = api.run("/pneumothorax_segmentation/PneumothoraxSegmentation/runs/1znusep6")
run_34_44 = api.run("/pneumothorax_segmentation/PneumothoraxSegmentation/runs/17u1gocc")
run_45_50 = api.run("/pneumothorax_segmentation/PneumothoraxSegmentation/runs/1cg3si8x")


df = pd.concat([run_0.history(), run_1_7.history(), run_8.history(), run_9_11.history(), run_12_21.history(), run_22.history(), run_23_32.history(), run_33.history(), run_34_44.history(), run_45_50.history()], ignore_index=True)
print(df)

# Plot Train and Validation IoU versus Epoch (for df)
train_iou = df['train_iou']
val_iou = df['val_iou']
train_loss = df['train_loss']
valid_loss = df['val_loss']
epoch = df['epoch']
learning_rate = df['started_lr']

plt.plot(epoch, train_iou, label='Train IoU')
plt.plot(epoch, val_iou, label='Validation IoU')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.title("IoU Against Epoch")
plt.legend()
plt.show()

plt.plot(epoch, learning_rate, label='Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title("Learning Rate Against Epoch")
plt.legend()
plt.show()

plt.plot(epoch, train_loss, label='Train Loss')
plt.plot(epoch, valid_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title("Loss Against Epoch")
plt.legend()
plt.show()

df.to_csv('z_csv/wandb_export_0_50.csv', index=False)