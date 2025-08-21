import numpy as np
from keras.layers import Input,Flatten,Dropout,Dense
from keras.models import Model
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.optimizers import adam_v2
from models_Res_ConvNet_trans import *

###############################################
# preprocess the data with "extract_features_with_meta_and_filter()" function before training
# call the resnet and ConvNeXt models as below in order to use
# make sure the data size is (n, 5000, 12)
# epoch = 60, batch_size= 64
#


############################################################
#resnet
############################################################
input_tensor = Input(shape=(1, 5000, 12), dtype='float32')
x = resnet_CNN_1(input_tensor,9,16)
x = resnet_CNN_1(x,7,32)
x = resnet_CNN_1(x,5,64)
x = resnet_CNN_1(x,3,128)
x = resnet_CNN_2(x)
x = Flatten()(x)
x = Dropout(0.1)(x) #0.1
output_tensor = Dense(1, activation='sigmoid',kernel_regularizer=regularizers.l2(0.01))(x)


resnet50 = Model(input_tensor, output_tensor)


def step_decay(epoch, lr):
    if epoch > 24:
        return lr
    else:
        return lr * np.math.exp(-0.0005)
    

lr_scheduler = LearningRateScheduler(step_decay)
opt = adam_v2.Adam(learning_rate=0.001, beta_1=0.5)
resnet50.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy']) 

############################################################
#ConvNeXt
############################################################
ConvNeXt_model=get_convnext_model()
opt = adam_v2.Adam(learning_rate=0.0001, beta_1=0.5) #learning_rate=0.0005
ConvNeXt_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
