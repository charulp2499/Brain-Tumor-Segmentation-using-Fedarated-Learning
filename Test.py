import nibabel as nib
import cv2
import gc
import os
import numpy as np
from glob import glob
import tensorflow as tf
from tqdm.auto import tqdm
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

from Codes.Model import UNET3D
from Codes.Dataloader import *
from Codes.Evalution_matrix import *
from Codes.ASWA import sum_scaled_weights
from Codes.SFAL import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#Load dataset for client 1
client_1_images_file = os.listdir("/kaggle/input/msd01-64/processed_64/images")
client_1_masks_file = os.listdir("/kaggle/input/msd01-64/processed_64/masks")
client_1_npy_images_path = [os.path.join("/kaggle/input/msd01-64/processed_64/images", img_file) for img_file in client_1_images_file[:]]
client_1_npy_masks_path = [os.path.join("/kaggle/input/msd01-64/processed_64/masks", mas_file) for mas_file in client_1_masks_file[:]]
# client_1_weight = "/kaggle/input/server/Communication_3_Client_1_Weights60 (1).h5"
client_1_npy_images_path.sort()
client_1_npy_masks_path.sort()

# Load dataset for client 2
client_2_images_file = os.listdir("/kaggle/input/pdgm-64/PDGM/image")
client_2_masks_file = os.listdir("/kaggle/input/pdgm-64/PDGM/mask")
client_2_npy_images_path = [os.path.join("/kaggle/input/pdgm-64/PDGM/image", img_file) for img_file in client_2_images_file[:]]
client_2_npy_masks_path = [os.path.join("/kaggle/input/pdgm-64/PDGM/mask", mas_file) for mas_file in client_2_masks_file[:]]
# client_2_weight = "/kaggle/input/iith-communication-2-clients-weights/Communication_2_Client_2_Weights60.h5"
client_2_npy_images_path.sort()
client_2_npy_masks_path.sort()

# Load dataset for client 3
client_3_images_file = os.listdir("/kaggle/input/msd01-64/processed_64/images")
client_3_masks_file = os.listdir("/kaggle/input/msd01-64/processed_64/masks")
client_3_npy_images_path = [os.path.join("/kaggle/input/msd01-64/processed_64/images", img_file) for img_file in client_3_images_file[:]]
client_3_npy_masks_path = [os.path.join("/kaggle/input/msd01-64/processed_64/masks",mas_file) for mas_file in client_3_masks_file[:]]
# client_3_weight = "/kaggle/input/iith-communication-2-clients-weights/Communication_2_Client_3_Weights60.h5"
client_3_npy_images_path.sort()
client_3_npy_masks_path.sort()


# Split data for each client
client_1_train_images, client_1_train_masks, client_1_valid_images, client_1_valid_masks, client_1_test_images, client_1_test_masks = split_train_valid_test_data(client_1_npy_images_path, client_1_npy_masks_path)
client_2_train_images, client_2_train_masks, client_2_valid_images, client_2_valid_masks, client_2_test_images, client_2_test_masks = split_train_valid_test_data(client_2_npy_images_path, client_2_npy_masks_path)
client_3_train_images, client_3_train_masks, client_3_valid_images, client_3_valid_masks, client_3_test_images, client_3_test_masks = split_train_valid_test_data(client_3_npy_images_path, client_3_npy_masks_path)
test_data = mini_batches_(client_1_test_images+client_2_test_images+client_3_test_images,
                                           client_1_test_masks+client_2_test_masks+client_3_test_masks,5)

#Server
Server=dict()
Server["model"] = UNET3D(1).call()
Server["model"].load_weights('/kaggle/input/server/server.h5')

dice_test = []
pre_test = []
batch_loss_test = []
se_test = []
spe_test = []
io_test = []
model = Server['model']

for batch_idx, (images, masks) in enumerate(tqdm(test_data)):
    predictions, logits = model(images)

    loss = dice_loss2(masks,logits)
    batch_loss_test.append(loss)

    dice_test.append(dice_coef(masks, logits))
    pre_test.append(precision(masks, logits))
    se_test.append(sensitivity(masks, logits))
    spe_test.append(specificity(masks, logits))
    io_test.append(iou(masks, logits))

batch_loss_test = np.array(batch_loss_test).mean()
dice_test = np.array(dice_test).mean()
pre_test = np.array(pre_test).mean()
se_test = np.array(se_test).mean()
spe_test = np.array(spe_test).mean()
io_test = np.array(io_test).mean()

print("\nLoss: {} , Dice Coeff: {}\n,Precision: {} Sensitivity: {} \n Specificity: {} , IOU: {}".format(batch_loss_test, dice_test, pre_test, se_test, spe_test, io_test))
                