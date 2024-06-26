import os
import numpy as np
from tqdm.auto import tqdm

def load_img(img_list):
    images=[]
    for i, image_name in enumerate(img_list):
        image = np.load(image_name)
        images.append(image)
    images = np.array(images)
    return(images)


def imageLoader(img_list,mask_list, batch_size):
    L = len(img_list)
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)

            X = load_img(img_list[batch_start:limit])
            Y = load_img(mask_list[batch_start:limit])
            Y = Y.astype(np.float32)
            yield (X,Y)
            batch_start += batch_size
            batch_end += batch_size

def mini_batches_( X, Y, batch_size=1):
    train_length = len(X)
    num_batches = int(np.floor(train_length / batch_size))
    batches = []
    for i in tqdm(range(num_batches)):
        batch_x = X[i * batch_size: i * batch_size + batch_size]
        batch_y = Y[i * batch_size:i * batch_size + batch_size]
        batch_x = load_img(batch_x)
        batch_y = load_img(batch_y)
        batch_x = batch_x.astype(np.float32)
        batch_y = batch_y.astype(np.float32)
        batches.append([batch_x, batch_y])
    return batches

def split_train_valid_test_data(images_path, masks_path, valid_percentage=0.1, test_percentage=0.15):
    num_samples = len(images_path)

    num_valid_samples = int(valid_percentage * num_samples)
    num_test_samples = int(test_percentage * num_samples)

    valid_images = images_path[:num_valid_samples]
    valid_masks = masks_path[:num_valid_samples]

    test_images = images_path[num_valid_samples:num_valid_samples + num_test_samples]
    test_masks = masks_path[num_valid_samples:num_valid_samples + num_test_samples]

    train_images = images_path[num_valid_samples + num_test_samples:]
    train_masks = masks_path[num_valid_samples + num_test_samples:]

    return train_images, train_masks, test_images, test_masks, valid_images, valid_masks