import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from genericpath import exists

TRAIN_VAL_RATIO = 0.2
BATCH_SIZE = 4
IMG_SIZE = (128,128)
SHUFFLE = True    
NUM_CLASSES = 3
RANDOM_SEED = 120
AUG = {
    'FLIP_H': 0.3,
    'FLIP_V': 0.6,
    'ROTATE_CW': 0.4,
    'ROTATE_CCW': 0.7,
    'TRANSLATE': [5, 0.3],
    'RAND_BRIGHTNESS': 0.5,
}

class DataLoader:
    def __init__(self, 
                 path, 
                 split_ratio=TRAIN_VAL_RATIO,
                 batch_size = BATCH_SIZE,
                 img_size = IMG_SIZE,
                 shuffle = SHUFFLE,
                 num_classes = NUM_CLASSES,
                 random_seed = RANDOM_SEED,
                 augment_conf = AUG):
        
        super(DataLoader, self).__init__()        
        self.path = path
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.random_seed = random_seed
        self.augment_conf = augment_conf

        assert exists(self.path), "Dataset file does not exist, please insert the right input filepath."

        print('Importing the datasets with the following parameters...')
        print('   Dataset path                    :', self.path)
        print('   Train-Val dataset ratio        :', self.split_ratio)

        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = self.read_dataset_path(subset='TrainVal')
        self.test_img_paths, self.test_mask_paths = self.read_dataset_path(subset='Test')

        assert len(self.train_img_paths) == len(self.train_mask_paths), "image length files do not match with masks lenth in the training data"
        assert len(self.val_img_paths) == len(self.val_mask_paths), "image length files do not match with masks lenth in the validation data"
        assert len(self.test_img_paths) == len(self.test_mask_paths), "image length files do not match with masks lenth in the testing data"

        print('num of training data: ', len(self.train_img_paths))
        print('num of validation data: ', len(self.val_img_paths))
        print('num of testing data: ', len(self.test_img_paths))


    def read_dataset_path(self, subset='TrainVal'):

        image_dir = os.path.join(self.path, subset, 'color')
        mask_dir = os.path.join(self.path, subset, 'label')
        
        image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.jpg')])
        mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir) if fname.endswith('.png')])
        
        if subset == 'TrainVal':
            print('\nSplitting training and validation sets...')
            train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
                image_paths, mask_paths, test_size=self.split_ratio, random_state=self.random_seed)
            return train_img_paths, val_img_paths, train_mask_paths, val_mask_paths
        elif subset == 'Test':
            return image_paths, mask_paths
        else:
            raise KeyError ("subset is wrong")

    def create_dataset(self):
        def load_image_mask(image_path, mask_path, image_size=(128, 128), num_classes=3):

            image = tf.io.read_file(image_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, image_size) / 255.0  
            
            mask = tf.io.read_file(mask_path)
            mask = tf.image.decode_png(mask, channels=3)
            mask = tf.image.resize(mask, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            is_black = tf.reduce_all(tf.equal(mask, [0, 0, 0]), axis=-1)
            is_white = tf.reduce_all(tf.equal(mask, [255, 255, 255]), axis=-1)
            is_red = tf.equal(mask[..., 0], 128)
            is_green = tf.equal(mask[..., 1], 128)

            red_count = tf.reduce_sum(tf.cast(is_red, tf.uint8))
            green_count = tf.reduce_sum(tf.cast(is_green, tf.uint8))

            def turn_white_red():
                return tf.logical_or(is_red, is_white), is_green

            def turn_white_green():
                return is_red, tf.logical_or(is_green, is_white)

            new_is_red, new_is_green = tf.cond(red_count > green_count, turn_white_red, turn_white_green)

            ch_1 = tf.cast(is_black, tf.uint8) * 0
            ch_2 = tf.cast(new_is_red, tf.uint8) * 1
            ch_3 = tf.cast(new_is_green, tf.uint8) * 2
            label = ch_1 + ch_2 + ch_3
            label = tf.reshape(label, (128,128,1))
            return image, label

        def augment(image, mask):
            
            if tf.random.uniform(()) <= self.augment_conf['FLIP_H']:
                image = tf.image.flip_left_right(image)
                mask = tf.image.flip_left_right(mask)
            
            if tf.random.uniform(()) <= self.augment_conf['FLIP_V']:
                image = tf.image.flip_up_down(image)
                mask = tf.image.flip_up_down(mask)

            if tf.random.uniform(()) <= self.augment_conf['ROTATE_CW']:
                image = tf.image.rot90(image)
                mask = tf.image.rot90(mask)
                rot_ccw = False
            else:
                rot_ccw = True
            
            if tf.random.uniform(()) <= self.augment_conf['ROTATE_CCW'] and rot_ccw == True:
                image = tf.image.rot90(image, k=3)
                mask = tf.image.rot90(mask, k=3)
            
            if tf.random.uniform(()) <= self.augment_conf['TRANSLATE'][1]:
                dx = tf.random.uniform([], -1*self.augment_conf['TRANSLATE'][0], 
                                       self.augment_conf['TRANSLATE'][0], dtype=tf.int32)
                dy = tf.random.uniform([], -1*self.augment_conf['TRANSLATE'][0], 
                                       self.augment_conf['TRANSLATE'][0], dtype=tf.int32)
                image = tf.roll(image, shift=[dx, dy], axis=[0, 1])
                mask = tf.roll(mask, shift=[dx, dy], axis=[0, 1])
                
            if tf.random.uniform(()) <= self.augment_conf['RAND_BRIGHTNESS']:      
                image = tf.image.random_brightness(image, max_delta=0.2)
            
            return image, mask
    
        def build_dataset(image_paths, mask_paths, augment_data=True):
            dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
            dataset = dataset.map(lambda img, mask: load_image_mask(img, mask, self.img_size, self.num_classes), num_parallel_calls=tf.data.AUTOTUNE)
            if augment_data:
                dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.cache().shuffle(1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            return dataset
        
        train_dataset = build_dataset(self.train_img_paths, self.train_mask_paths, augment_data=True)
        val_dataset = build_dataset(self.val_img_paths, self.val_mask_paths, augment_data=False)
        test_dataset = build_dataset(self.test_img_paths, self.test_img_paths, augment_data=False)
        
        return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    dataset_path = "Dataset"
    data_loader = DataLoader(dataset_path)
    train_dataset, val_dataset, test_dataset = data_loader.create_dataset()