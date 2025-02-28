
from genericpath import exists
from glob import glob
from sklearn.model_selection import train_test_split
from PIL import Image

import os
import random
import tensorflow as tf
import numpy as np
import cv2


RANDOM_SEED = 100
BATCH_SIZE = 2
TRAIN_VAL_RATIO = 0.7
random.seed(RANDOM_SEED)

    # def __init__(self, path, train_val_ratio=TRAIN_VAL_RATIO, batch_size=BATCH_SIZE):
    #     super(DatasetLoader, self).__init__()
    #     self.path = path
    #     self.train_val_ratio = train_val_ratio
    #     self.batch_size = batch_size

def read_dataset(path):

    assert exists(path), "Dataset file does not exist, please insert the right input filepath."

    print('Importing the datasets with the following parameters...')
    print('   Dataset path                    :', path)
    print('   Train-Val dataset ratio        :', TRAIN_VAL_RATIO)

    trainval_img_paths = np.array(glob(os.path.join(path, "TrainVal/color/*.jpg")))
    trainval_img_paths = sorted(trainval_img_paths)
    trainval_label_path = np.array(glob(os.path.join(path, "TrainVal/label/*.png")))
    trainval_label_path = sorted(trainval_label_path)

    test_img_paths = np.array(glob(os.path.join(path, "Test/color/*.jpg")))
    test_img_paths = sorted(test_img_paths)
    test_label_paths = np.array(glob(os.path.join(path, "Test/label/*.png")))
    test_label_paths = sorted(test_label_paths)

    num_trainval_files = len(trainval_img_paths)
    num_test_files = len(test_img_paths)

    assert num_trainval_files == len(trainval_label_path), "number of trainval img is not same as its labels"
    assert num_test_files == len(test_label_paths), "number of test img is not same as its labels"

    print('\nSplitting training and validation sets...')

    train_img_paths, val_img_paths, train_label_paths, val_label_paths = train_test_split(
        trainval_img_paths, trainval_label_path, test_size=(1-TRAIN_VAL_RATIO), random_state=RANDOM_SEED)
    
    # print('HUFTTT', type(self.train_img_paths))
    print('num of training data: ', len(train_img_paths))
    print('num of validation data: ', len(val_img_paths))
    print('num of testing data: ', num_test_files)

    train_ds = tf.data.Dataset.from_tensor_slices((train_img_paths, train_label_paths))
    val_ds = tf.data.Dataset.from_tensor_slices((val_img_paths, val_label_paths))
    test_ds = tf.data.Dataset.from_tensor_slices((test_img_paths, test_label_paths))

    return train_ds, val_ds, test_ds        


def prepare_dataset(train_ds, val_ds, test_ds, flip=None, rotate=None, translate=None):
    # def augment_wrapper(img, label):
    #     flipped, rot, trans = img_augment(img, label, flip=flip, rotate=rotate, translate=translate)
    #     augmented_ds = tf.data.Dataset.from_tensors((img, label))
    #     augmented_ds = augmented_ds.concatenate(tf.data.Dataset.from_tensors(flipped))
    #     augmented_ds = augmented_ds.concatenate(tf.data.Dataset.from_tensors(rot))
    #     augmented_ds = augmented_ds.concatenate(tf.data.Dataset.from_tensors(trans))
    #     return augmented_ds
    
    # train_ds, val_ds, test_ds = read_dataset()
    train_ds = train_ds.map(lambda img, label: img_augment(img, label, flip=flip, rotate=rotate, translate=translate),
                        num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache().shuffle(50).repeat(3).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.cache().shuffle(50).map(img_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).repeat(3).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    test_ds = test_ds.map(img_preprocessing, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    train_ds = iter(train_ds)
    val_ds = iter(val_ds)
    test_ds = iter(test_ds)

    return train_ds, val_ds, test_ds
    
    # def __len__(self):
    #     return len(self.train_img_paths)*4 // self.batch_size
    
    # def __getitem__(self, index):
    #     return self.data[index]


def img_preprocessing(img_path, label_path, shape=(128,128)):

    # img = cv2.imread(str(img_path))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = np.array(img, dtype=np.float32)/255.0

    # label = cv2.imread(label_path)
    # label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

    # is_black = np.all(label == [0,0,0], axis=-1)
    # is_white = np.all(label == [255,255,255], axis=-1)
    # is_red = (label[...,0] == 128)
    # is_green = (label[...,2] == 128)

    # if is_red.any():
    #     is_red |= is_white
    # else:
    #     is_green |= is_white

    # ch_1 = is_black.astype(np.uint8)
    # ch_2 = is_red.astype(np.uint8)
    # ch_3 = is_green.astype(np.uint8)

    # label = np.stack([ch_1, ch_2, ch_3], axis=-1)

    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, shape)
    img = img /255.0
    img = tf.cast(img, dtype=tf.float32)

    label = tf.io.read_file(label_path)
    label = tf.io.decode_png(label, channels=3)
    label = tf.image.resize(label, shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    is_black = tf.reduce_all(tf.equal(label, [0, 0, 0]), axis=-1)
    is_white = tf.reduce_all(tf.equal(label, [255, 255, 255]), axis=-1)
    is_red = tf.equal(label[..., 0], 128)
    is_green = tf.equal(label[..., 1], 128)
    
    red_count = tf.reduce_sum(tf.cast(is_red, tf.int32))
    green_count = tf.reduce_sum(tf.cast(is_green, tf.int32))

    # Apply the logic
    def turn_white_red():
        return tf.logical_or(is_red, is_white), is_green

    def turn_white_green():
        return is_red, tf.logical_or(is_green, is_white)

    new_is_red, new_is_green = tf.cond(red_count > green_count, turn_white_red, turn_white_green)

    ch_1 = tf.cast(is_black, tf.uint8)
    ch_2 = tf.cast(new_is_red, tf.uint8)
    ch_3 = tf.cast(new_is_green, tf.uint8)

    
    label = tf.stack([ch_1, ch_2, ch_3], axis=-1)


    return img, label

def img_augment(img, label, flip=None, rotate=None, translate=None):
    def _image_flip(img, label, mode):

        if mode == 'horizontal':
            img = tf.image.flip_left_right(img)
            label = tf.image.flip_left_right(label)
        elif mode == 'vertical':
            img = tf.image.flip_up_down(img)
            label = tf.image.flip_up_down(label)
        else:
            print(f'mode is not recognized: {mode}. Rederence: horizontal or vertical. No flipping algorithm is performed')

        return img, label

    def _image_rotate(img, label, mode):

        if mode == 'ccw':
            img = tf.image.rot90(img, k=1)
            label = tf.image.rot90(label, k=1)
        elif mode == 'cw':
            img = tf.image.rot90(img, k=3)
            label = tf.image.rot90(label, k=3)
        else:
            print(f'mode is not recognized: {mode}. Reference: cw or ccw. No rotate algorithm is performed')

        return img, label
    
    def _image_translate(img, label, mode):

        if mode:
            dx = tf.random.uniform([], -5, 5, dtype=tf.int32)
            dy = tf.random.uniform([], -5, 5, dtype=tf.int32)
            img = tf.roll(img, shift=[dx, dy], axis=[0, 1])
            label = tf.roll(label, shift=[dx, dy], axis=[0, 1])
        else:
            print(f'mode is not recognized: {mode}. Reference: Yes. No translation algorithm is performed')

        return img, label
    
    # flip_img, flip_label = _image_flip(img, label, flip)
    # rot_img, rot_label = _image_rotate(img, label, rotate)
    # trans_img, trans_label = _image_translate(img, label, translate)
    # return (flip_img, flip_label), (rot_img, rot_label), (trans_img, trans_label)

    img, label = tf.py_function(img_preprocessing, [img, label], Tout=[tf.float32, tf.uint8])

    if tf.random.uniform(()) > 0.5:
        img, label = _image_flip(img, label, flip)
    if tf.random.uniform(()) > 0.5:
        img, label = _image_rotate(img, label, rotate)
    if tf.random.uniform(()) > 0.5:
        img, label = _image_translate(img, label, translate)
    return img, label
    
if __name__ == "__main__":
    train_ds, val_ds, test_ds = read_dataset('./Dataset/Dataset')
    a, b, c = prepare_dataset(train_ds, val_ds, test_ds, flip='horizontal',
                                                          rotate='ccw',
                                                          translate=30)