import tensorflow as tf
import pydot
import numpy as np
from matplotlib import pyplot as plt
import keras
import keras.utils
from keras.layers.core import Dropout, Lambda
from keras import utils as np_utils
from keras.utils.np_utils import to_categorical
from keras.datasets import cifar10
from keras.utils import np_utils
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import sys
import random
import time
import warnings
from PIL import Image
from keras.preprocessing import image
import tensorflow_addons as tfa

####################################################
#### VERİ OKUMA YAPILDIĞI VARSAYILIR
data1 = []  # normal  -  X_train        Eğitim için input datası
data2 = []  # segmente  -Y_train        Eğitim için output datası
tutucu = []

# verilerin tip fönüşümü float32 türüne dönüştürülüyor
data1 = np.array(data1, dtype="float32")
data2 = np.array(data2, dtype="float32")

# Eğitimdeki giriş datamız olacak görüntü boyutlarımız tutuluyor.
img_width = data1.shape[1]
img_height = data1.shape[2]
img_channels = 1

# tekrarlama işlemi ile datamız augmentation işlemi için 4 parametreli hale getiriliyor.
print(data1.shape)  # (64, 224, 224)
data1 = np.repeat(data1[..., np.newaxis], 1, -1)
print(data1.shape)  # (64, 224, 224, 1)

print(data2.shape)  # (64, 224, 224)
data2 = np.repeat(data2[..., np.newaxis], 1, -1)
print(data2.shape)  # (64, 224, 224, 1)

#### TEST PREPROCESS
data3 = []  # normal  -  X_Test
data4 = []  # segmente  -Y_Test

data3 = np.array(data3, dtype="float32")
data4 = np.array(data4, dtype="float32")

print(data3.shape)  # (64, 224, 224)
data3 = np.repeat(data3[..., np.newaxis], 1, -1)
print(data3.shape)  # (64, 224, 224, 1)

print(data4.shape)  # (64, 224, 224)
data4 = np.repeat(data4[..., np.newaxis], 1, -1)
print(data4.shape)  # (64, 224, 224, 1)

####################################################
#### VERİ ZENGİNLEŞTİRME
# Veri zenginleştirme yapılacaksa Görüntü ve Maske Generator Objeleri Oluşturuluyor
image_datagen = image.ImageDataGenerator(
    shear_range=0.5,
    rotation_range=50,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode="nearest",
)
mask_datagen = image.ImageDataGenerator(
    shear_range=0.5,
    rotation_range=50,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode="nearest",
)

BATCH_SIZE = (
    4  # Batch size data augmentation eklenince datagen.flow fonksiyonunda kullanılıyor
)
seed = 16  # Sabit tohum belirleniyor

X_train = data1  # Eğitim datanızı X_Train ile ifade ediyoruz
Y_train = data2  # Segmentasyon datanızı Y_Train ile ifade ediyoruz


image_datagen.fit(X_train, augment=True, seed=seed)
mask_datagen.fit(Y_train, augment=True, seed=seed)

x = image_datagen.flow(
    X_train, batch_size=BATCH_SIZE, shuffle=True, seed=seed
)  # numpy X_train dizisi döndürülür.
y = mask_datagen.flow(Y_train, batch_size=BATCH_SIZE, shuffle=True, seed=seed)


# Görüntü ve maske oluşturucular için aynı tohumu bir araya getiriyoruz ki böylece birbirine uysunlar

image_datagen.fit(X_train[: int(X_train.shape[0] * 1)], augment=True, seed=seed)
mask_datagen.fit(Y_train[: int(Y_train.shape[0] * 1)], augment=True, seed=seed)

x = image_datagen.flow(
    X_train[: int(X_train.shape[0] * 1)], batch_size=BATCH_SIZE, shuffle=True, seed=seed
)
y = mask_datagen.flow(
    Y_train[: int(Y_train.shape[0] * 1)], batch_size=BATCH_SIZE, shuffle=True, seed=seed
)

############## Doğrulama datası için Görüntü ve Maske Generator Objeleri Oluşturuluyor
image_datagen_val = image.ImageDataGenerator()
mask_datagen_val = image.ImageDataGenerator()

image_datagen_val.fit(X_train[int(X_train.shape[0] * 0.9) :], augment=True, seed=seed)
mask_datagen_val.fit(Y_train[int(Y_train.shape[0] * 0.9) :], augment=True, seed=seed)

x_val = image_datagen_val.flow(
    X_train[int(X_train.shape[0] * 0.9) :],
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=seed,
)
y_val = mask_datagen_val.flow(
    Y_train[int(Y_train.shape[0] * 0.9) :],
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=seed,
)

imshow(x.next()[0].astype(np.uint8))
plt.show()
imshow(np.squeeze(y.next()[0].astype(np.uint8)))
plt.show()

train_generator = zip(x, y)
val_generator = zip(x_val, y_val)

####################################################
#### UNET BÖLÜTLEME MODELİ OLUŞTURMA

inputs = tf.keras.layers.Input((img_width, img_height, 1))  # input layer
# s=tf.keras.Layers.Lambda(lambda x : x/255.)(inputs)
filterNum = 8

c1 = tf.keras.layers.Conv2D(
    filterNum, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
)(inputs)
c1 = tf.keras.layers.BatchNormalization()(c1)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(
    filterNum, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
)(c1)
c1 = tf.keras.layers.BatchNormalization()(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)  # Avf pooling

c2 = tf.keras.layers.Conv2D(
    filterNum * 2,
    (3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
)(p1)
c2 = tf.keras.layers.BatchNormalization()(c2)
# c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(
    filterNum * 2,
    (3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
)(c2)
c2 = tf.keras.layers.BatchNormalization()(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(
    filterNum * 4,
    (3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
)(p2)
c3 = tf.keras.layers.BatchNormalization()(c3)
# c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(
    filterNum * 4,
    (3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
)(c3)
c3 = tf.keras.layers.BatchNormalization()(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

c4 = tf.keras.layers.Conv2D(
    filterNum * 8,
    (3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
)(p3)
c4 = tf.keras.layers.BatchNormalization()(c4)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(
    filterNum * 8,
    (3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
)(c4)
c4 = tf.keras.layers.BatchNormalization()(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

c5 = tf.keras.layers.Conv2D(
    filterNum * 16,
    (3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
)(p4)
c5 = tf.keras.layers.BatchNormalization()(c5)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(
    filterNum * 16,
    (3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
)(c5)
c5 = tf.keras.layers.BatchNormalization()(c5)
p5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c5)

c6 = tf.keras.layers.Conv2D(
    filterNum * 32,
    (3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
)(p5)
c6 = tf.keras.layers.BatchNormalization()(c6)
# c6 = tf.keras.layers.Dropout(0.3)(c6)
c6 = tf.keras.layers.Conv2D(
    filterNum * 32,
    (3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
)(c6)
c6 = tf.keras.layers.BatchNormalization()(c6)

# Expansive path
u7 = tf.keras.layers.Conv2DTranspose(
    filterNum * 16, (2, 2), strides=(2, 2), padding="same"
)(c6)
u7 = tf.keras.layers.concatenate([u7, c5])
c7 = tf.keras.layers.Conv2D(
    filterNum * 16,
    (3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
)(u7)
c7 = tf.keras.layers.BatchNormalization()(c7)
# c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(
    filterNum * 16,
    (3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
)(c7)
c7 = tf.keras.layers.BatchNormalization()(c7)

u8 = tf.keras.layers.Conv2DTranspose(
    filterNum * 8, (2, 2), strides=(2, 2), padding="same"
)(c7)
u8 = tf.keras.layers.concatenate([u8, c4])
c8 = tf.keras.layers.Conv2D(
    filterNum * 8,
    (3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
)(u8)
c8 = tf.keras.layers.BatchNormalization()(c8)
c8 = tf.keras.layers.Dropout(0.2)(c8)
c8 = tf.keras.layers.Conv2D(
    filterNum * 8,
    (3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
)(c8)
c8 = tf.keras.layers.BatchNormalization()(c8)

u9 = tf.keras.layers.Conv2DTranspose(
    filterNum * 4, (2, 2), strides=(2, 2), padding="same"
)(c8)
u9 = tf.keras.layers.concatenate([u9, c3])
c9 = tf.keras.layers.Conv2D(
    filterNum * 4,
    (3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
)(u9)
c9 = tf.keras.layers.BatchNormalization()(c9)
c9 = tf.keras.layers.Dropout(0.2)(c9)
c9 = tf.keras.layers.Conv2D(
    filterNum * 4,
    (3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
)(c9)
c9 = tf.keras.layers.BatchNormalization()(c9)

u10 = tf.keras.layers.Conv2DTranspose(
    filterNum * 2, (2, 2), strides=(2, 2), padding="same"
)(c9)
u10 = tf.keras.layers.concatenate([u10, c2])
c10 = tf.keras.layers.Conv2D(
    filterNum * 2,
    (3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
)(u10)
c10 = tf.keras.layers.BatchNormalization()(c10)
# c10 = tf.keras.layers.Dropout(0.1)(c10)
c10 = tf.keras.layers.Conv2D(
    filterNum * 2,
    (3, 3),
    activation="relu",
    kernel_initializer="he_normal",
    padding="same",
)(c10)
c10 = tf.keras.layers.BatchNormalization()(c10)

u11 = tf.keras.layers.Conv2DTranspose(
    filterNum, (2, 2), strides=(2, 2), padding="same"
)(c10)
u11 = tf.keras.layers.concatenate([u11, c1], axis=3)
c11 = tf.keras.layers.Conv2D(
    filterNum, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
)(u11)
c11 = tf.keras.layers.BatchNormalization()(c11)
c11 = tf.keras.layers.Dropout(0.1)(c11)
c11 = tf.keras.layers.Conv2D(
    filterNum, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same"
)(c11)
c11 = tf.keras.layers.BatchNormalization()(c11)

outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(
    c11
)  # çıktı bir kanal olacak

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
# tf.keras.optimizers.ADAM(lr = 0.003)

####################################################
#### UNET BÖLÜTLEME MODELİ EĞİTME

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.MeanIoU(num_classes=2)],
)  # callbacks
model.summary()

results = model.fit(train_generator, steps_per_epoch=len(x) // BATCH_SIZE, epochs=20)

model.save("model1")


"""
model.fit_generator(train_generator, validation_data=val_generator, validation_steps=10, steps_per_epoch=50,
                            epochs=10)
                            
                                 
"""

""" 
model.fit(x=data1, 
          y=data2,
          batch_size=4,
          epochs=10)  

"""
####################################################
#### UNET BÖLÜTLEME MODELİ DEĞERLENDİRME

model = keras.models.load_model("model1")

# Eğitilmiş olan modelimiz için tahmin datası oluşturuluyor
BS = 4
start = time.time()
print("[INFO] model degerlendiriliyor...")
tahminler = model.predict(
    data3, batch_size=BS
)  # her data1 görüntüsüne karşılık bir tahmin dizisi içermektedir.( nummpy dizisi içerisinde)
print(time.time() - start)

# Tüm tahmin datası, gerçek test dataları ile İoU metriği üzerinde karşılaştırılmalı olarak test edilir.
for x in range(len(data4)):
    data = data4[x]
    tahminlerN = tahminler[x]
    tahminlerN = np.where(tahminlerN >= 0.8, 1, 0)
    for i in range(640):
        for j in range(640):
            if np.logical_and(data[i][j] == 1, tahminlerN[i][j] == 1):
                k = k + 1
            if np.logical_and(data[i][j] == 1, tahminlerN[i][j] == 0):
                l = l + 1
            if np.logical_and(data[i][j] == 0, tahminlerN[i][j] == 1):
                l = l + 1

iou = k / (k + l)

print(iou)
