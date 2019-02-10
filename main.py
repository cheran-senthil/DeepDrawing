import glob

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from keras import backend
from keras.layers import (Activation, BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, Input, Reshape, ZeroPadding2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from PIL import Image


def load():
    filelist = glob.glob("./images/*")
    x = [np.array(Image.open(fname))[:, :, :3] for fname in filelist]
    return np.array(x)


# HyperParameters
optimizer = Adam(0.0001, 0.5, decay=1e-6)
loss = 'binary_crossentropy'
batch_size = 84
epochs = 1000
s_interval = 50

# Image PreProcessing
real_images = load()
shape = real_images.shape
img_rows = shape[0]
img_column = shape[1]
img_channels = shape[2]
if backend.image_data_format() == 'channels_first':
    real_images = np.reshape(real_images,
                             (shape[0], shape[3], shape[1], shape[2]))
else:
    real_images = np.reshape(real_images,
                             (shape[0], shape[1], shape[2], shape[3]))

img_shape = real_images.shape[1:]
real_images = real_images.astype('float32')
real_images /= 255

# Create Generator
generator = Sequential()
generator.add(Dense(128 * 64 * 36, activation="relu", input_dim=100))
generator.add(Reshape((36, 64, 128)))
generator.add(UpSampling2D())
generator.add(Conv2D(64, kernel_size=3, padding="same"))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Activation("relu"))
generator.add(UpSampling2D())
generator.add(Conv2D(3, kernel_size=3, padding="same"))
generator.add(BatchNormalization(momentum=0.8))
generator.add(Activation("sigmoid"))

# Create Discriminator
discriminator = Sequential()

discriminator.add(
    Conv2D(
        32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.25))
discriminator.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
discriminator.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
discriminator.add(BatchNormalization(momentum=0.8))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.25))
discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
discriminator.add(BatchNormalization(momentum=0.8))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.25))
discriminator.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
discriminator.add(BatchNormalization(momentum=0.8))
discriminator.add(LeakyReLU(alpha=0.2))
discriminator.add(Dropout(0.25))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.summary()

discriminator.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

# Create Combined Model
noise = Input(shape=(100, ))
img = generator(noise)

discriminator.trainable = False
valid = discriminator(img)

combined = Model(noise, valid)
combined.compile(loss=loss, optimizer=optimizer)

# Training
fake = np.zeros((batch_size, 1))
valid = np.ones((batch_size, 1))
pbar = trange(epochs)
for epoch in pbar:
    # Train Discriminator
    idx = np.random.randint(0, real_images.shape[0], batch_size)
    imgs = real_images[idx]
    n = np.random.normal(0, 1, (batch_size, 100))
    gen_imgs = generator.predict(n)
    #    imgs = np.append(imgs, gen_imgs, axis=0)
    labels = np.append(valid, fake)
    d_loss = discriminator.train_on_batch(imgs, valid)
    d_loss = discriminator.train_on_batch(gen_imgs, fake)

    # Train Generator
    g_loss = combined.train_on_batch(n, valid)

    pbar.set_description("[D loss: %.3f, acc.: %.2f%%] [G loss: %.3f]" %
                         (d_loss[0], 100 * d_loss[1], g_loss))

    if epoch % s_interval == 0:
        r, c = 5, 5
        n = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = generator.predict(n)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("gen_images/%d.png" % epoch)
        plt.close()
