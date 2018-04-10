""" based on Wei Ji, https://www.kaggle.com/weiji14/yet-another-keras-u-net-data-augmentation/notebook """

##
# Import all the necessary libraries
import os
import datetime
import glob
import random
import sys
import pickle

import matplotlib.pyplot as plt
import skimage.transform                              #Used for resize function
# from skimage.morphology import label                  #Used for Run-Length-Encoding RLE to create final submission

import numpy as np
import pandas as pd

import keras
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv2DTranspose
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.merge import add, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import multi_gpu_model, plot_model
from keras import backend as K
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split

from tqdm import tqdm

# os.environ['CUDA_VISIBLE_DEVICES']="0"

print('Python       :', sys.version.split('\n')[0])
print('Numpy        :', np.__version__)
print('Skimage      :', skimage.__version__)
print('Scikit-learn :', sklearn.__version__)
print('Keras        :', keras.__version__)
print('Tensorflow   :', tf.__version__)
print('CUDA_VISIBLE_DEVICES={}'.format(os.environ['CUDA_VISIBLE_DEVICES']))



""" set parameters """

# Set number of GPUs
num_gpus = 1   #defaults to 1 if one-GPU or one-CPU. If 4 GPUs, set to 4.

# Set height (y-axis length) and width (x-axis length) to train model on
img_height, img_width = (256, 256)  #Default to (256,266), use (None,None) if you do not want to resize imgs

##
""" prepare data """

with open('./data/data_train.pickle', 'rb') as f:
    data_train = pickle.load(f)
with open('./data/data_test.pickle', 'rb') as f:
    data_test = pickle.load(f)

def prepare_data(data_dict):
    n = len(data_dict)
    X = np.zeros((n, img_height, img_width, 3), dtype='uint8')
    Y = np.zeros((n, img_height, img_width, 1), dtype='bool')
    for i, img_id in tqdm(enumerate(data_dict)):
        image_org = data_dict[img_id]['image']
        X[i, :, :, :] = skimage.transform.resize(image_org, (img_height, img_width), mode='constant', preserve_range=True)
        if ('mask' in data_dict[img_id]) and data_dict[img_id]['mask'].size > 0:
            mask_org = (data_dict[img_id]['mask'] > 0)
            Y[i, :, :, 0] = skimage.transform.resize(mask_org, (img_height, img_width), mode='constant', preserve_range=True)
    return X, Y

X_train, Y_train = prepare_data(data_train)
X_test, _ = prepare_data(data_test)

# plot a sample
def plot_image_mask(image, mask_true=None, mask_pred=None):

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    mask_rgb = image*0.0
    if mask_pred is not None:
        mask_rgb[:, :, 0] = mask_pred[:, :, 0]
    if mask_true is not None:
        mask_rgb[:, :, 1] = mask_true[:, :, 0]
    plt.imshow((mask_rgb*255).astype('uint8'))
    plt.axis('off')

i = np.random.randint(X_train.shape[0])
plot_image_mask(X_train[i], Y_train[i], Y_train[i])

##
""" build model """

# Set seed values
seed = 42
random.seed = seed
np.random.seed(seed=seed)

# Design our model architecture here
def keras_model(img_width=256, img_height=256):
    '''
    Modified from https://keunwoochoi.wordpress.com/2017/10/11/u-net-on-keras-2-0/
    '''
    n_ch_exps = [4, 5, 6, 7, 8, 9]  # the n-th deep channel's exponent i.e. 2**n 16,32,64,128,256
    k_size = (3, 3)  # size of filter kernel
    k_init = 'he_normal'  # kernel initializer

    if K.image_data_format() == 'channels_first':
        ch_axis = 1
        input_shape = (3, img_width, img_height)
    elif K.image_data_format() == 'channels_last':
        ch_axis = 3
        input_shape = (img_width, img_height, 3)

    inp = Input(shape=input_shape)
    encodeds = []

    # encoder
    enc = inp
    print(n_ch_exps)
    for l_idx, n_ch in enumerate(n_ch_exps):
        enc = Conv2D(filters=2 ** n_ch, kernel_size=k_size, activation='relu', padding='same',
                     kernel_initializer=k_init)(enc)
        enc = Dropout(0.1 * l_idx, )(enc)
        enc = Conv2D(filters=2 ** n_ch, kernel_size=k_size, activation='relu', padding='same',
                     kernel_initializer=k_init)(enc)
        encodeds.append(enc)
        # print(l_idx, enc)
        if n_ch < n_ch_exps[-1]:  # do not run max pooling on the last encoding/downsampling step
            enc = MaxPooling2D(pool_size=(2, 2))(enc)

    # decoder
    dec = enc
    print(n_ch_exps[::-1][1:])
    decoder_n_chs = n_ch_exps[::-1][1:]
    for l_idx, n_ch in enumerate(decoder_n_chs):
        l_idx_rev = len(n_ch_exps) - l_idx - 2  #
        dec = Conv2DTranspose(filters=2 ** n_ch, kernel_size=k_size, strides=(2, 2), activation='relu', padding='same',
                              kernel_initializer=k_init)(dec)
        dec = concatenate([dec, encodeds[l_idx_rev]], axis=ch_axis)
        dec = Conv2D(filters=2 ** n_ch, kernel_size=k_size, activation='relu', padding='same',
                     kernel_initializer=k_init)(dec)
        dec = Dropout(0.1 * l_idx)(dec)
        dec = Conv2D(filters=2 ** n_ch, kernel_size=k_size, activation='relu', padding='same',
                     kernel_initializer=k_init)(dec)

    outp = Conv2DTranspose(filters=1, kernel_size=k_size, activation='sigmoid', padding='same',
                           kernel_initializer='glorot_normal')(dec)

    model = Model(inputs=[inp], outputs=[outp])

    return model

# Custom IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# Custom loss function
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


# Set some model compile parameters
optimizer = 'adam'
loss      = bce_dice_loss
metrics   = [mean_iou]

# Compile our model
model = keras_model(img_width=img_width, img_height=img_height)
model.summary()

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)



##
""" run model """


# Runtime data augmentation
def get_train_test_augmented(X_data=X_train, Y_data=Y_train, validation_split=0.25, batch_size=32, seed=seed):
    X_train, X_test, Y_train, Y_test = train_test_split(X_data,
                                                        Y_data,
                                                        train_size=1 - validation_split,
                                                        test_size=validation_split,
                                                        random_state=seed)

    # Image data generator distortion options
    data_gen_args = dict(rotation_range=45.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='reflect')  # use 'constant'??

    # Train data, provide the same seed and keyword arguments to the fit and flow methods
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)
    X_datagen.fit(X_train, augment=True, seed=seed)
    Y_datagen.fit(Y_train, augment=True, seed=seed)
    X_train_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seed)
    Y_train_augmented = Y_datagen.flow(Y_train, batch_size=batch_size, shuffle=True, seed=seed)

    # Test data, no data augmentation, but we create a generator anyway
    X_datagen_val = ImageDataGenerator()
    Y_datagen_val = ImageDataGenerator()
    X_datagen_val.fit(X_test, augment=True, seed=seed)
    Y_datagen_val.fit(Y_test, augment=True, seed=seed)
    X_test_augmented = X_datagen_val.flow(X_test, batch_size=batch_size, shuffle=True, seed=seed)
    Y_test_augmented = Y_datagen_val.flow(Y_test, batch_size=batch_size, shuffle=True, seed=seed)

    # combine generators into one which yields image and masks
    train_generator = zip(X_train_augmented, Y_train_augmented)
    test_generator = zip(X_test_augmented, Y_test_augmented)

    return train_generator, test_generator


##
# Runtime custom callbacks
# %% https://github.com/deepsense-ai/intel-ai-webinar-neural-networks/blob/master/live_loss_plot.py
# Fixed code to enable non-flat loss plots on keras model.fit_generator()
import matplotlib.pyplot as plt
from keras.callbacks import Callback
from IPython.display import clear_output


# from matplotlib.ticker import FormatStrFormatter

def translate_metric(x):
    translations = {'acc': "Accuracy", 'loss': "Log-loss (cost function)"}
    if x in translations:
        return translations[x]
    else:
        return x


class PlotLosses(Callback):
    def __init__(self, figsize=None, path_save_train_log_fig=None):
        super(PlotLosses, self).__init__()
        self.figsize = figsize
        self.path_save_train_log_fig = path_save_train_log_fig

    def on_train_begin(self, logs={}):

        self.base_metrics = [metric for metric in self.params['metrics'] if not metric.startswith('val_')]
        self.logs = []
        plt.figure(figsize=self.figsize)

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs.copy())

        plt.clf()
        clear_output(wait=True)

        for metric_id, metric in enumerate(self.base_metrics):
            plt.subplot(1, len(self.base_metrics), metric_id + 1)

            plt.plot(range(1, len(self.logs) + 1),
                     [log[metric] for log in self.logs],
                     label="training")
            if self.params['do_validation']:
                plt.plot(range(1, len(self.logs) + 1),
                         [log['val_' + metric] for log in self.logs],
                         label="validation")
            plt.title(translate_metric(metric))
            plt.xlabel('epoch')
            plt.legend(loc='center left')

        plt.tight_layout()
        # plt.show()
        if (self.path_save_train_log_fig is not None) and epoch%5==0:
            plt.savefig(self.path_save_train_log_fig)


##
""" train """

# Finally train the model!!
batch_size = 16

train_generator, test_generator = get_train_test_augmented(X_data=X_train, Y_data=Y_train,
                                                           validation_split=0.11, batch_size=batch_size)

model_out = model
time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
path_unet_weight = './unet_logs'
if not os.path.isdir(path_unet_weight):
        os.makedirs(path_unet_weight)
path_save_train_log_fig = os.path.join(path_unet_weight, 'unet_weight_{}.png'.format(time_str))

plot_losses = PlotLosses(figsize=(16, 4), path_save_train_log_fig = path_save_train_log_fig)

plt.ioff()
model.fit_generator(train_generator, validation_data=test_generator, validation_steps=batch_size/2,
                    steps_per_epoch=len(X_train)/(batch_size*2), epochs=100,
                    callbacks=[plot_losses])

model_out.save_weights(os.path.join(path_unet_weight, 'unet_weight_{}.h5'.format(time_str)))
plt.savefig(os.path.join(path_unet_weight, 'unet_weight_{}.png'.format(time_str)))
plt.close('all')
plt.ion()

##
""" test """

model_loaded = keras_model(img_width=img_width, img_height=img_height)

path_unet_weight = './unet_logs'
model_loaded.load_weights(os.path.join(path_unet_weight, 'unet_weight_20180408_025630.h5'))

model_loaded.compile(optimizer=optimizer, loss=loss, metrics=metrics)
##

i = np.random.randint(X_train.shape[0])
x = X_train[i:i+1, :, :, ]
y = Y_train[i:i+1, :, :, ]
y_hat = model_loaded.predict(x, verbose=1)>0.2


plot_image_mask(image=x[0], mask_true=y[0], mask_pred=y_hat[0])

##
i = np.random.randint(X_test.shape[0])
# x = X_test
x = X_train
y_true = Y_train
y_hat = model_loaded.predict(x, verbose=1)


data_detection = {}
time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DIR_DETECTION_RESULT = './detection_result'
if not(os.path.exists(DIR_DETECTION_RESULT)):
    os.mkdir(DIR_DETECTION_RESULT)
path_cur_detection_result = os.path.join(DIR_DETECTION_RESULT, time_str)
os.mkdir(path_cur_detection_result)

plt.ioff()
for i in range(x.shape[0]):
    plot_image_mask(image=x[i], mask_pred=y_hat[i], mask_true=y_true[i])
    plt.savefig( os.path.join(path_cur_detection_result, '{}.png'.format(i)))
    plt.close()
plt.ion()
##


plt.close('all')


