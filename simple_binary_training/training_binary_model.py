import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization

###extra things
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
    strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
else:
    strategy = tf.distribute.get_strategy()
    
    
tf.test.is_built_with_cuda()
tf.config.list_physical_devices('GPU')


def load_data(csv_file, image_dir):
    df = pd.read_csv(csv_file)
    df['FILENAME'] = df['FILENAME'].apply(lambda x: os.path.join(image_dir, x))
    df['class_bin'] = df['class_bin'].astype(str)
    # Map integer labels to string categories
    #label_map = {i: f'category_{i}' for i in range(10)}
    #df['label'] = df['label'].astype(int).map(label_map)
    return df


# Function to create train and validation data generators
def create_datasets(df, batch_size, img_height, img_width):
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_datagen = ImageDataGenerator(rescale=1./255)
    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='FILENAME',
        y_col='class_bin',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='sparse',  # Use 'categorical' for one-hot encoding
        shuffle=True
    )

    valid_generator = valid_datagen.flow_from_dataframe(
        dataframe=valid_df,
        x_col='FILENAME',
        y_col='class_bin',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='sparse',  # Use 'categorical' for one-hot encoding
        shuffle=False
    )

    return train_generator, valid_generator


def build_model():
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = Sequential()
        # Convolutional Layer
        model.add(Conv2D(filters=512, kernel_size=(3, 3), input_shape=(256, 256, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        # Pooling Layer
        model.add(MaxPool2D(pool_size=(2, 2)))
        # Dropout Layer
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
        #              metrics= METRICS)
        #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=METRICS)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        

    return model



df = load_data('bestgalaxies_random_60k_TESTVALSAMPLEONLY.csv',
               '/luidhy_docker/astrodados/DELVE_MORPHOLOGY_DATA/CONTROL_SAMPLE_CNN/DOMINGUEZ_galaxies/DOMINGUEZ2018_images_rgb')
train_gen, valid_gen = create_datasets(df, batch_size=100, img_height=256, img_width=256)
model = build_model()



early_stopping = EarlyStopping(monitor='val_loss', patience=35, restore_best_weights=True)
history = model.fit(train_gen, validation_data=valid_gen, epochs=50,callbacks=[early_stopping])
#model.save('teste.h5')


#making some  plots

model.save('model_results/teste1.h5')


# In[ ]:


history_frame = pd.DataFrame(history.history)
history_frame.info()
history_frame.to_csv("model_results/teste1.csv",index=False)


# In[ ]:


#acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']
#loss = history.history['loss']
#val_loss = history.history['val_loss']

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(10, 10))

# Plot training accuracy
plt.plot(epochs, acc, label='Training Accuracy', color='red')
plt.plot(epochs, val_acc, label='Validation Accuracy', color='blue')
plt.tick_params(labelsize=16,direction='in',top=True,right=True,width='1',size=6)
plt.locator_params(axis="x", nbins=15)
plt.locator_params(axis="y", nbins=15)
plt.title('Training and Validation Accuracy')
plt.legend(fontsize=15)
plt.xlabel('Epochs',fontsize=20)
plt.ylabel('Accuracy',fontsize=20)
plt.tight_layout()
plt.savefig('model_results/teste1_Accuracy.png',facecolor='white',dpi=200)
plt.figure(figsize=(10, 10))

# Plot training loss
plt.plot(epochs, loss, label='Training Loss', color='red')
plt.plot(epochs, val_loss, label='Validation Loss', color='blue')
plt.tick_params(labelsize=16,direction='in',top=True,right=True,width='1',size=6)
plt.locator_params(axis="x", nbins=15)
plt.locator_params(axis="y", nbins=15)
plt.title('Training and Validation Loss')
plt.legend(fontsize=15)
plt.xlabel('Epochs',fontsize=20)
plt.ylabel('Loss',fontsize=20)
plt.tight_layout()
plt.savefig('model_results/teste1__Loss.png',facecolor='white',dpi=200)