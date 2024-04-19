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

epochs = 50
num_comm = 3
batch_size = 5
LR = 0.00001
optim = tf.keras.optimizers.Adam(LR)

# Client 1
client_1 ={}
client_1["original_data"] = mini_batches_(client_1_train_images, client_1_train_masks,batch_size)
client_1["model"] = UNET3D(1).call()
client_1["optimizer"] = tf.keras.optimizers.Adam(LR)
# client_1["model"].load_weights(client_1_weight)
client_1["test_data"] = mini_batches_(client_1_valid_images, client_1_valid_masks,batch_size)

# Client 2
client_2 ={}
client_2["original_data"] = mini_batches_(client_2_train_images, client_2_train_masks,batch_size)
client_2["model"] = UNET3D(1).call()
client_2["optimizer"] = tf.keras.optimizers.Adam(LR)
# client_2["model"].load_weights(client_2_weight)
client_2["test_data"] = mini_batches_(client_2_valid_images, client_2_valid_masks,batch_size)

#Client 3
client_3 ={}
client_3["original_data"] = mini_batches_(client_3_train_images, client_3_train_masks,batch_size)
client_3["model"] = UNET3D(1).call()
client_3["optimizer"] = tf.keras.optimizers.Adam(LR)
# client_3["model"].load_weights(client_3_weight)
client_3["test_data"] = mini_batches_(client_3_valid_images, client_3_valid_masks,batch_size)

test_data = mini_batches_(client_1_test_images+client_2_test_images+client_3_test_images,
                                           client_1_test_masks+client_2_test_masks+client_3_test_masks,batch_size)

#Server
Server=dict()
Server["model"] = UNET3D(1).call()

Data_Scales=list()
Total_data = len(client_1_npy_images_path) + len(client_2_npy_images_path) + len(client_3_npy_images_path)
Data_Scales.append(len(client_1_npy_images_path)/ Total_data)
Data_Scales.append(len(client_2_npy_images_path)/Total_data)
Data_Scales.append(len(client_3_npy_images_path)/Total_data)

clients = [client_1,client_2,client_3]
metrics = [iou, dice_coef, precision, sensitivity, specificity]
past_model = None
print(f"Total Communication Round: {num_comm}")

for t in range(num_comm):
    print('Communication round: ', t+1)

    for epoch in range(epochs):
        print(f'\n--------------------------------- Epoch: {epoch + 1}---------------------------------')
        i = 1
        local_weight_list = list()
        for client in clients:
            print(f'Client {i}')
            model = client['model']
            optimizer = client["optimizer"]
            model.set_weights(client["model"].get_weights())
            dice = []
            average_weights = []
            pre = []
            batch_loss = []
            se = []
            spe = []
            io = []

            for batch_idx, (images, masks) in enumerate(tqdm(client['original_data'])):
                pro2, _ = Server['model'](images)
                with tf.GradientTape() as tape:
                    pro1, logits = model(images)
                    if 'past_model' not in client:
                        pro3 = tf.zeros_like(pro1)
                    else:
                        past_model = client['past_model']
                        pro3, past = past_model(images)

                    loss_combined = SFAL(y_true=masks, y_pred=logits, prev=pro3, pres=pro1, serv=pro2, temperature=0.5, dice_weight=0.5)
                    loss = loss_combined
                    batch_loss.append(loss)

                    dice.append(dice_coef(masks, logits))
                    pre.append(precision(masks, logits))
                    se.append(sensitivity(masks, logits))
                    spe.append(specificity(masks, logits))
                    io.append(iou(masks, logits))

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            local_weight_list.append(model.get_weights())
            batch_loss = np.array(batch_loss).mean()
            dice = np.array(dice).mean()
            pre = np.array(pre).mean()
            se = np.array(se).mean()
            spe = np.array(spe).mean()
            io = np.array(io).mean()
            i += 1
            
            print("Combined Loss (D/C): {} | Dice Coeff: {}  |\n Precision: {} Sensitivity: {} | \n Specificity: {} , IOU: {} | \n\n".format(batch_loss, dice, pre, se, spe, io))
            
            if epoch == epochs-1:                
                print("---------------------------------Client Test Data---------------------------------")
                dice_test = []
                pre_test = []
                batch_loss_test = []
                se_test = []
                spe_test = []
                io_test = []
                model = client['model']

                for batch_idx, (images, masks) in enumerate(tqdm(client['test_data'])):
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
                print("----------------------------------------------------------------------------------")
                
        server_model = Server["model"]
        average_weights = sum_scaled_weights(local_weight_list,Data_Scales)
        server_model.set_weights(average_weights)
        
        for client in clients:
            client['past_model'] = server_model
            client['model'] = server_model


