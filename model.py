import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *

class DownBlock(Layer):
    def __init__(self, filters, use_maxpool=True, name='encoder'):
        super().__init__(name=name)
        self.filters = filters
        self.use_maxpool = use_maxpool
    
        self.conv_1 = Conv2D(self.filters, 3, padding= 'same', kernel_initializer = 'he_normal')
        self.bn_1 = BatchNormalization()
        self.act_1 = LeakyReLU()
        self.conv_2 = Conv2D(self.filters, 3, padding= 'same', kernel_initializer = 'he_normal')
        self.bn_2 = BatchNormalization()
        self.act_2 = LeakyReLU()
        if self.use_maxpool == True:
            self.maxpool = MaxPooling2D(strides= (2,2))
    
    def call(self, inputs, training=False):
        out = self.conv_1(inputs)
        out = self.bn_1(out, training=training)
        out = self.act_1(out)
        out = self.conv_2(out)
        out = self.bn_2(out, training=training)
        out = self.act_2(out)
        if self.use_maxpool == True:
            out = self.maxpool(out)
            return out, inputs
        else:
            return out
    
class UpBlock(Layer):
    def __init__(self, filters, name='decoder'):
        super().__init__(name=name)
        self.filters = filters
        self.up = UpSampling2D()
        self.concat = Concatenate(axis=3)
        self.conv_1 = Conv2D(self.filters, 3, padding='same', kernel_initializer = 'he_normal')
        self.bn_1 = BatchNormalization()
        self.act_1 = LeakyReLU()
        self.conv_2 = Conv2D(self.filters, 3, padding='same', kernel_initializer = 'he_normal')
        self.bn_2 = BatchNormalization()
        self.act_2 = LeakyReLU()
    
    def call(self, inputs, skip_layer, training=False):
        x = self.up(inputs)
        out = self.concat([x, skip_layer])
        out = self.conv_1(out)
        out = self.bn_1(out, training=training)
        out = self.act_1(out)
        out = self.conv_2(out)
        out = self.bn_2(out, training=training)
        out = self.act_2(out)
        return out
    
class UNet(models.Model):
    def __init__(self, num_class, dropout, filters=[64,128,256,512, 1024], name='UNet'):
        super().__init__(name=name)
        self.filters = filters
        self.num_class = num_class
        self.dropout = dropout

        self.down_1 = DownBlock(filters[0], name='encoder_1')
        self.down_2 = DownBlock(filters[1], name='encoder_2')
        self.down_3 = DownBlock(filters[2], name='encoder_3')
        self.down_4 = DownBlock(filters[3], name='encoder_4')
        self.down_5 = DownBlock(filters[4], name='encoder_5', use_maxpool=False)

        self.up_1 = UpBlock(filters[3], name='decoder_1')
        self.up_2 = UpBlock(filters[2], name='decoder_2')
        self.up_3 = UpBlock(filters[1], name='decoder_3')
        self.up_4 = UpBlock(filters[0], name='decoder_4')
        
        self.dropout = Dropout(dropout)
        self.conv = Conv2D(num_class, 1, activation='softmax')

    def call(self, inputs, training=False):
        out, skip1 = self.down_1(inputs, training=training)
        out, skip2 = self.down_2(out, training=training)
        out, skip3 = self.down_3(out, training=training)
        out, skip4 = self.down_4(out, training=training)
        out = self.down_5(out, training=training)

        out = self.up_1(out, skip4, training=training)
        out = self.up_2(out, skip3, training=training)
        out = self.up_3(out, skip2, training=training)
        out = self.up_4(out, skip1, training=training)

        out = self.dropout(out)
        out = self.conv(out)
        return out

# if __name__ == '__main__':
#     model = UNet(3, 0.2)
#     inputs = Input([128,128,3])
#     out = model(inputs, training=True)
#     a = tf.keras.Model(inputs=inputs, outputs=out)
#     a.summary()

