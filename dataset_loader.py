
from genericpath import exists
from glob import glob
from sklearn.model_selection import train_test_split
from PIL import Image

import os
import random
import tensorflow as tf
import numpy as np

RANDOM_SEED = 100
BATCH_SIZE = 16
TRAIN_VAL_RATIO = 0.7
random.seed(RANDOM_SEED)

class DatasetLoader():
    def __init__(self, path, train_val_ratio=TRAIN_VAL_RATIO, batch_size=BATCH_SIZE):
        self.path = path
        self.train_val_ratio = train_val_ratio
        self.batch_size = batch_size

    def read_dataset(self):

        assert exists(self.path), "Dataset file does not exist, please insert the right input filepath."

        print('Importing the datasets with the following parameters...')
        print('   Dataset path                    :', self.path)
        print('   Train-Val dataset ratio        :', self.train_val_ratio)

        trainval_img_paths = glob(os.path.join(self.path, "TrainVal/color/*.jpg"))
        trainval_label_path = glob(os.path.join(self.path, "TrainVal/label/*.png"))

        test_img_paths = glob(os.path.join(self.path, "Test/color/*.jpg"))
        test_label_paths = glob(os.path.join(self.path, "Test/label/*.png"))

        num_trainval_files = len(trainval_img_paths)
        num_test_files = len(test_img_paths)

        assert num_trainval_files == len(trainval_label_path), "number of trainval img is not same as its labels"
        assert num_test_files == len(test_label_paths), "number of test img is not same as its labels"

        print('\nSplitting training and validation sets...')

        train_img_paths, val_img_paths, train_label_paths, val_label_paths = train_test_split(
            trainval_img_paths, trainval_label_path, test_size=(1-self.train_val_ratio), random_state=RANDOM_SEED)
        
        print('num of training data: ', len(train_img_paths))
        print('num of validation data: ', len(val_img_paths))
        print('num of testing data: ', num_test_files)

        train_ds = tf.data.Dataset.from_tensor_slices((train_img_paths, train_label_paths))
        val_ds = tf.data.Dataset.from_tensor_slices((val_img_paths, val_label_paths))
        test_ds = tf.data.Dataset.from_tensor_slices((test_img_paths, test_label_paths))

        return train_ds, val_ds, test_ds
    
    def img_preprocessing(self, img_path, label_path, shape=(128,128)):
        # def _resize(img_path, shape):
        #     img_path = img_path.decode("utf-8")
        #     img = Image.open(img_path)
        #     img = img.resize(shape)
        #     img = np.array(img, dtype=np.float32)/255.0   # mau normalisasi 0-1 atau pake mean-var?
        #     return img
        
        # img = tf.py_function(func=_resize, inp=[img_path], Tout=tf.float32)
        # label = tf.py_function(func=_resize, inp=[label_path], Tout=tf.float32)

        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, shape)
        img = img / 255.0  # mau normalisasi 0-1 atau pake mean-var?

        label = tf.io.read_file(label_path)
        label = tf.image.decode_png(label, channels=1)
        label = tf.image.resize(label, shape)
        label = label /255.0

        return img, label
    
    def img_augment(self, img, label, flip=None, rotate=None, translate=None):
        def _image_flip(img, label, mode):
            # if mode == 'horizontal':
            #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
            #     label = label.transpose(Image.FLIP_LEFT_RIGHT)
            # elif mode == 'vertical':
            #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
            #     label = label.transpose(Image.FLIP_TOP_BOTTOM)

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
            # if mode == 'cw':
            #     img = img.transpose(Image.TRANSPOSE)
            #     label = label.transpose(Image.TRANSPOSE)
            # elif mode == 'ccw':
            #     img = img.transpose(Image.TRANSVERSE)
            #     label = label.transpose(Image.TRANSVERSE)

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
            # if mode :
            #     img = img.transform(img.size, Image.AFFINE, (1,0,-30,0,1,0))
            #     label = label.transform(label.size, Image.AFFINE, (1,0,-30,0,1,0))

            if mode:
                dx = tf.random.uniform([], -5, 5, dtype=tf.int32)
                dy = tf.random.uniform([], -5, 5, dtype=tf.int32)
                img = tf.roll(img, shift=[dx, dy], axis=[0, 1])
                label = tf.roll(label, shift=[dx, dy], axis=[0, 1])
            else:
                print(f'mode is not recognized: {mode}. Reference: Yes. No translation algorithm is performed')

            return img, label
        
        # def _augment(img, label):

        #     img, label = tf.py_function(func=_image_flip, inp=[img, label, flip], Tout=tf.float32)
        #     img, label = tf.py_function(func=_image_rotate, inp=[img, label, flip], Tout=tf.float32)
        #     img, label = tf.py_function(func=_image_translate, inp=[img, label, flip], Tout=tf.float32)

        #     return img, label
        
        flip_img, flip_label = _image_flip(img, label, flip)
        rot_img, rot_label = _image_rotate(img, label, rotate)
        trans_img, trans_label = _image_translate(img, label, translate)
        
        # img, label = _image_flip(img, label, flip)
        # img, label = _image_rotate(img, label, rotate)
        # img, label = _image_translate(img, label, translate)
        return (flip_img, flip_label), (rot_img, rot_label), (trans_img, trans_label)
    

    def prepare_dataset(self, flip=None, rotate=None, translate=None):
        def augment_wrapper(img, label):
            flipped, rot, trans = self.img_augment(img, label, flip=flip, rotate=rotate, translate=translate)
            return tf.data.Dataset.from_tensors((img, label)).concatenate(tf.data.Dataset.from_tensors(flipped)).concatenate(tf.data.Dataset.from_tensors(rot)).concatenate(tf.data.Dataset.from_tensors(trans))
        
        train_ds, val_ds, test_ds = self.read_dataset()
        train_ds = train_ds.map(self.img_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.flat_map(augment_wrapper)
        # train_ds = train_ds.map(lambda img, label: self.img_augment(img, label, flip=flip, rotate=rotate, translate=translate),
        #                     num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = train_ds.shuffle(1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        # train_ds = iter(train_ds)

        val_ds = val_ds.map(self.img_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.shuffle(1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        # val_ds = iter(val_ds)

        test_ds = test_ds.map(self.img_preprocessing, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        # test_ds = iter(test_ds)

        return train_ds, val_ds, test_ds

    
if __name__ == "__main__":
    datasetLoader = DatasetLoader('./Dataset')
    train_ds, val_ds, test_ds = datasetLoader.prepare_dataset(flip='horizontal',
                                                          rotate='ccw',
                                                          translate=30)