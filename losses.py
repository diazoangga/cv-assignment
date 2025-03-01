import tensorflow as tf
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *



class CustomLoss(Loss):
    def __init__(self, loss_conf):
        super(CustomLoss, self).__init__()
        self.dice_loss_coeff = loss_conf['DICE_COEFF']
        self.iou_loss_coeff = loss_conf['IOU_COEFF']
        self.focal_loss_coeff = loss_conf['FOCAL_LOSS_COEFF']
        self.focal_loss = SparseCategoricalCrossentropy()
    
    def call(self, y_true, y_pred):
        def dice_loss(y_true, y_pred, smooth=1):
            # print(y_pred.shape, y_true.shape)
            y_true = tf.cast(y_true, dtype=tf.float32)
            numerator = 2*tf.reduce_sum(y_true * y_pred) + smooth
            denominator = tf.reduce_sum(y_true + y_pred) + smooth

            return 1-(numerator/denominator)

        def iou_loss(y_true, y_pred, smooth=1):
            y_true = tf.cast(y_true, dtype=tf.float32)
            intersection = tf.reduce_sum(y_true * y_pred)
            union = tf.reduce_sum(y_true + y_pred) - intersection

            num = intersection + smooth
            den = union + smooth

            return 1-(num/den)

        return self.dice_loss_coeff*dice_loss(y_true, y_pred) + \
                self.iou_loss_coeff*iou_loss(y_true, y_pred) + \
                self.focal_loss_coeff*self.focal_loss(y_true, y_pred)

if __name__ == '__main__':
    LOSS_CONF = {'dice_loss': 0.2,
                 'iou_loss': 0.8,
                 'focal_loss': 0}
    loss = CustomLoss(LOSS_CONF)
    a = tf.ones([4,3])
    b = tf.random.normal([4,3])*0.1

    print(loss(a,b))
