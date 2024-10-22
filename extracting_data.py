import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
from PIL import Image
import os
import time


def main():
    input_dir="../Data/input/train/dicom_files/"
    output_dir="../Data-Output"
    # os.mkdir(output_dir)

    # stats waala part
    meta_df=pd.DataFrame(columns=["DICOM","Sex","Age","Pneumothorax","View Position","Patient ID"])

    train_data=pd.read_csv("../Data/input/train/train-rle.csv")
    for index,row in train_data.drop_duplicates(subset="ImageId").iterrows():
        print(index)
        
        image_id=row["ImageId"]
        pneumo_detected=0 if row[" EncodedPixels"].strip()=="-1" else 1

        # 1: DICOM to PNG
        ds=pydicom.dcmread(input_dir+image_id+".dcm")
        img_array=Image.fromarray(ds.pixel_array)
        save_path=f'{output_dir}/png_files/{image_id}.png'
        img_array.save(save_path)

        # 2: Meta Data Row
        new_row = {"DICOM":image_id, "Sex":ds.PatientSex, "Age":ds.PatientAge, "Pneumothorax":pneumo_detected, "View Position":ds.ViewPosition,"Patient ID":ds.PatientID}
        meta_df = meta_df._append(new_row, ignore_index=True)

        # print(img_array.size)
        final_mask=np.zeros(img_array.size)

        selected_rows=train_data[train_data["ImageId"]==image_id]

        # 3: Mask Generation
        for i, r in selected_rows.iterrows():
            rle=r[" EncodedPixels"]
            mask=rle2mask(rle,img_array.size[0],img_array.size[1])
            rotated_mask=np.rot90(mask,k=-1)
            flipped_mask=np.fliplr(rotated_mask)
            final_mask+=flipped_mask
        final_mask[final_mask >= 255] = 255

        plt.imsave(f'{output_dir}/mask_files/{image_id}_mask.png',final_mask,cmap='gray')
        
        # 4: Mask Layers
        if pneumo_detected:
            fig, ax = plt.subplots()
            ax.imshow(img_array, cmap='gray')
            ax.imshow(final_mask, cmap='Reds', alpha=0.2)
            ax.axis('off')
            plt.savefig(f'{output_dir}/mask_layers/{image_id}.png', bbox_inches='tight', pad_inches=0)
            plt.close

    meta_df.to_csv(f'{output_dir}/metadata.csv',index=False)

def rle2mask(rle, width=1024, height=1024):
    mask = np.zeros(width* height)
    if rle.strip() == '-1':
        return mask.reshape(width, height)
    array = np.asarray([int(x) for x in rle.strip().split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)

st = time.time()
main()
et = time.time()
print("time elapsed:", et-st, "seconds")