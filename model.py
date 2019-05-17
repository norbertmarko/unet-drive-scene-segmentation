from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2DTranspose, concatenate
from tensorflow.keras.layers import Input, MaxPooling2D, Dropout
from tensorflow.keras.models import Model #API

def conv_block(input_tensor, n_filter, kernel=(3, 3), padding='same', initializer="he_normal"):
    x = Conv2D(n_filter, kernel, padding=padding, kernel_initializer=initializer)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, kernel, padding=padding, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def deconv_block(input_tensor, residual, n_filter, kernel=(3, 3), strides=(2, 2), padding='same'):
    y = Conv2DTranspose(n_filter, kernel, strides, padding)(input_tensor)
    y = concatenate([y, residual], axis=3)
    y = conv_block(y, n_filter)
    return y

# NETWORK - n_classes is the desired number of classes, filters are fixed
def Unet(input_height, input_width, n_classes=4, filters=64):

    # Downsampling
    input_layer = Input(shape=(input_height, input_width, 3), name='input')

    conv_1 = conv_block(input_layer, filters)
    conv_1_out = MaxPooling2D(pool_size=(2, 2))(conv_1)

    conv_2 = conv_block(conv_1_out, filters*2)
    conv_2_out = MaxPooling2D(pool_size=(2, 2))(conv_2)

    conv_3 = conv_block(conv_2_out, filters*4)
    conv_3_out = MaxPooling2D(pool_size=(2, 2))(conv_3)

    conv_4 = conv_block(conv_3_out, filters*8)
    conv_4_out = MaxPooling2D(pool_size=(2, 2))(conv_4)
    conv_4_drop = Dropout(0.5)(conv_4_out)

    conv_5 = conv_block(conv_4_drop, filters*16)
    conv_5_drop = Dropout(0.5)(conv_5)

    # Upsampling
    deconv_1 = deconv_block(conv_5_drop, conv_4, filters*8)
    deconv_1_drop = Dropout(0.5)(deconv_1)

    deconv_2 = deconv_block(deconv_1_drop, conv_3, filters*4)
    deconv_2_drop = Dropout(0.5)(deconv_2)

    deconv_3 = deconv_block(deconv_2_drop, conv_2, filters*2)
    deconv_3 = deconv_block(deconv_3, conv_1, filters)

    # Output - mapping each 64-component feature vector to number of classes
    output = Conv2D(n_classes, (1, 1))(deconv_3)
    output = BatchNormalization()(output)
    output = Activation("softmax")(output)

    # embed into functional API
    model = Model(inputs=input_layer, outputs=output, name="Unet")
    return model
