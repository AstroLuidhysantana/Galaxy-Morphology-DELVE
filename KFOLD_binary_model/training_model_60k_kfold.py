import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
else:
    strategy = tf.distribute.get_strategy()

def load_data(csv_file, image_dir):
    df = pd.read_csv(csv_file)
    df['FILENAME'] = df['FILENAME'].apply(lambda x: os.path.join(image_dir, x))
    df['class_bin'] = df['class_bin'].astype(str)
    return df

def create_datasets(df, train_index, val_index, batch_size, img_height, img_width):
    train_df = df.iloc[train_index]
    valid_df = df.iloc[val_index]
    
    train_datagen = ImageDataGenerator(rescale=1./255)
    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='FILENAME',
        y_col='class_bin',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=True
    )

    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=valid_df,
        x_col='FILENAME',
        y_col='class_bin',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False
    )

    return train_generator, valid_generator

def build_model():
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = Sequential()
        model.add(Conv2D(filters=512, kernel_size=(3, 3), input_shape=(256, 256, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(2, activation='softmax'))

        #METRICS = [
        #    'accuracy',
        #    tf.keras.metrics.Precision(name='precision'),
        #    tf.keras.metrics.Recall(name='recall')
        #]
        #model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        #              loss='categorical_crossentropy',
        #              metrics=METRICS)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model

df = load_data('bestgalaxies_random_60k_TESTVALSAMPLEONLY.csv',
               '/luidhy_docker/astrodados/DELVE_MORPHOLOGY_DATA/CONTROL_SAMPLE_CNN/DOMINGUEZ_galaxies/DOMINGUEZ2018_images_rgb')

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1

for train_index, val_index in kf.split(df):
    print(f'Training fold {fold_no}...')
    
    train_gen, valid_gen = create_datasets(df, train_index, val_index, batch_size=128, img_height=256, img_width=256)
    model = build_model()

    history = model.fit(train_gen, validation_data=valid_gen, epochs=50)
    
    model.save(f'model_results/model_fold_{fold_no}.h5')

    history_frame = pd.DataFrame(history.history)
    history_frame.to_csv(f'model_results/history_fold_{fold_no}.csv', index=False)
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure(figsize=(10, 10))
    plt.plot(epochs, acc, label='Training Accuracy', color='red')
    plt.plot(epochs, val_acc, label='Validation Accuracy', color='blue')
    plt.tick_params(labelsize=16, direction='in', top=True, right=True, width='1', size=6)
    plt.locator_params(axis="x", nbins=15)
    plt.locator_params(axis="y", nbins=15)
    plt.title('Training and Validation Accuracy')
    plt.legend(fontsize=15)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.tight_layout()
    plt.savefig(f'model_results/accuracy_fold_{fold_no}.png', facecolor='white', dpi=200)
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.plot(epochs, loss, label='Training Loss', color='red')
    plt.plot(epochs, val_loss, label='Validation Loss', color='blue')
    plt.tick_params(labelsize=16, direction='in', top=True, right=True, width='1', size=6)
    plt.locator_params(axis="x", nbins=15)
    plt.locator_params(axis="y", nbins=15)
    plt.title('Training and Validation Loss')
    plt.legend(fontsize=15)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.tight_layout()
    plt.savefig(f'model_results/loss_fold_{fold_no}.png', facecolor='white', dpi=200)
    plt.close()
    
    fold_no += 1
