import os
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
print('TensorFlow version:', tf.__version__)

DATA_PATH = "/content/supervisely_person_clean_2667_img"

IMAGES_DIR = os.path.join(DATA_PATH, "images")
MASKS_DIR = os.path.join(DATA_PATH, "masks")

image_files = sorted(os.listdir(IMAGES_DIR))
mask_files = sorted(os.listdir(MASKS_DIR))

print("Images:", len(image_files))
print("Masks:", len(mask_files))
TARGET_SIZE = (192, 192)

def load_image(path):
    img = imageio.imread(path)

    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)

    if img.shape[-1] == 4:
        img = img[..., :3]

    img = tf.image.resize(img, TARGET_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    return img


def load_mask(path):
    mask = imageio.imread(path)

    if mask.ndim == 2:
        mask = mask[..., None]

    mask = tf.image.resize(mask, TARGET_SIZE, method="nearest")

    if mask.shape[-1] > 1:
        mask = tf.reduce_mean(mask, axis=-1, keepdims=True)

    mask = tf.cast(mask, tf.float32) / 255.0
    return mask

def dataset_generator(img_files, mask_files):

    for img_name, mask_name in zip(img_files, mask_files):

        img_path = os.path.join(IMAGES_DIR, img_name)
        mask_path = os.path.join(MASKS_DIR, mask_name)

        img = load_image(img_path)
        mask = load_mask(mask_path)

        yield img, mask
def dice_loss(y_true, y_pred, smooth=1.0):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    return 1.0 - (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )
def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    d = dice_loss(y_true, y_pred)
    return bce + d

def dice_coef(y_true, y_pred, smooth=1.0):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )

def iou_coef(y_true, y_pred, smooth=1.0):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

    return (intersection + smooth) / (union + smooth)

def conv_block(x, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x
from sklearn.model_selection import train_test_split

train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    image_files,
    mask_files,
    test_size=0.2,
    random_state=42
)
BATCH_SIZE = 6

train_ds = tf.data.Dataset.from_generator(
    lambda: dataset_generator(train_imgs, train_masks),
    output_signature=(
        tf.TensorSpec(shape=(192,192,3), dtype=tf.float32),
        tf.TensorSpec(shape=(192,192,1), dtype=tf.float32)
    )
)

val_ds = tf.data.Dataset.from_generator(
    lambda: dataset_generator(val_imgs, val_masks),
    output_signature=(
        tf.TensorSpec(shape=(192,192,3), dtype=tf.float32),
        tf.TensorSpec(shape=(192,192,1), dtype=tf.float32)
    )
)

train_ds = train_ds.cache().shuffle(1000).repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
def unet_model(input_shape=(192, 192, 3)):
    inputs = tf.keras.layers.Input(shape=input_shape)

    c1 = conv_block(inputs, 32)
    p1 = tf.keras.layers.MaxPool2D(2)(c1)

    c2 = conv_block(p1, 64)
    p2 = tf.keras.layers.MaxPool2D(2)(c2)

    c3 = conv_block(p2, 128)
    p3 = tf.keras.layers.MaxPool2D(2)(c3)

    c4 = conv_block(p3, 256)
    p4 = tf.keras.layers.MaxPool2D(2)(c4)

    bn = conv_block(p4, 256)

    u1 = tf.keras.layers.Conv2DTranspose(256, 2, strides=2, padding='same')(bn)
    u1 = tf.keras.layers.Concatenate()([u1, c4])
    c5 = conv_block(u1, 256)

    u2 = tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c5)
    u2 = tf.keras.layers.Concatenate()([u2, c3])
    c6 = conv_block(u2, 128)

    u3 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c6)
    u3 = tf.keras.layers.Concatenate()([u3, c2])
    c7 = conv_block(u3, 64)

    u4 = tf.keras.layers.Conv2DTranspose(32, 2, strides=2, padding='same')(c7)
    u4 = tf.keras.layers.Concatenate()([u4, c1])
    c8 = conv_block(u4, 32)

    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', dtype='float32')(c8)

    return tf.keras.models.Model(inputs, outputs)

model = unet_model((192, 192, 3))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=bce_dice_loss,
    metrics=[dice_coef, iou_coef]
)

model.summary()
NUM_EPOCHS = 30
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "/content/drive/MyDrive/best_model.keras",
    monitor="val_dice_coef",
    save_best_only=True,
    mode="max"
)
from tensorflow.keras.callbacks import CSVLogger

csv_logger = CSVLogger("/content/drive/MyDrive/history.csv", append=True)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=NUM_EPOCHS,
    steps_per_epoch=300,
    validation_steps=80,
    callbacks=[checkpoint, csv_logger]
)
model = tf.keras.models.load_model(
    "/content/drive/MyDrive/best_model.keras",
    custom_objects={
        "bce_dice_loss": bce_dice_loss,
        "dice_coef": dice_coef,
        "iou_coef": iou_coef
    }
)
cm = confusion_matrix(y_true_all, y_pred_all)
TN, FP, FN, TP = cm.ravel()

precision = TP / (TP + FP + 1e-7)
recall = TP / (TP + FN + 1e-7)
f1 = 2 * precision * recall / (precision + recall + 1e-7)
accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-7)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Accuracy:", accuracy)
sample_ds = tf.data.Dataset.from_generator(
    lambda: dataset_generator(train_imgs[:5], train_masks[:5]),
    output_signature=(
        tf.TensorSpec(shape=(192,192,3), dtype=tf.float32),
        tf.TensorSpec(shape=(192,192,1), dtype=tf.float32)
    )
).batch(5)

for imgs, masks in sample_ds.take(1):
    for i in range(len(imgs)):

        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        plt.imshow(imgs[i])
        plt.title("Image")
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.imshow(masks[i,...,0], cmap="gray")
        plt.title("Mask")
        plt.axis("off")

        plt.show()