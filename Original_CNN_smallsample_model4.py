#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4"
#os.environ["CUDA_VISIBLE_DEVICES"]="6,7"
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
#from keras.preprocessing import image
from tensorflow import keras
#%matplotlib inline
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
#import tensorflow as tf
import numpy as np
import csv
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

###to run the new architecture
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
###to run on multigpu
if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
    strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
else:
    strategy = tf.distribute.get_strategy()
    
    
tf.test.is_built_with_cuda()
tf.config.list_physical_devices('GPU')


# In[2]:


#####setting  the folders location

TRAINING_ESPIRAL_DIR = "/luidhy_docker/astrodados/DELVE_MORPHOLOGY_DATA/analysis_code_morphology/CNN_testes_simples/default_CNN_smallsample/data_to_work_v5_balance/train/spiral"
VALIDATION_ESPIRAL_DIR = "/luidhy_docker/astrodados/DELVE_MORPHOLOGY_DATA/analysis_code_morphology/CNN_testes_simples/default_CNN_smallsample/data_to_work_v5_balance/val/spiral"
TESTING_ESPIRAL_DIR = "/luidhy_docker/astrodados/DELVE_MORPHOLOGY_DATA/analysis_code_morphology/CNN_testes_simples/default_CNN_smallsample/data_to_work_v5_balance/test/spiral"

TRAINING_ELIPTICA_DIR = "/luidhy_docker/astrodados/DELVE_MORPHOLOGY_DATA/analysis_code_morphology/CNN_testes_simples/default_CNN_smallsample/data_to_work_v5_balance/train/eliptical"
VALIDATION_ELIPTICA_DIR = "/luidhy_docker/astrodados/DELVE_MORPHOLOGY_DATA/analysis_code_morphology/CNN_testes_simples/default_CNN_smallsample/data_to_work_v5_balance/val/eliptical"
TESTING_ELIPTICA_DIR = "/luidhy_docker/astrodados/DELVE_MORPHOLOGY_DATA/analysis_code_morphology/CNN_testes_simples/default_CNN_smallsample/data_to_work_v5_balance/test/eliptical"

print("Counting files in each folder....")
print("For Spiral galaxies (LTGs)")
print("to train:", len(os.listdir(TRAINING_ESPIRAL_DIR)))
print("to validation:",len(os.listdir(VALIDATION_ESPIRAL_DIR)))
print("to test:", len(os.listdir(TESTING_ESPIRAL_DIR)))

print("For Eliptical galaxies ETGs")
print("to train:", len(os.listdir(TRAINING_ELIPTICA_DIR)))
print("to validation:", len(os.listdir(VALIDATION_ELIPTICA_DIR)))
print("to test:", len(os.listdir(TESTING_ELIPTICA_DIR)))


    

#####defining the model modelo 2 de cabeça para baixo
with strategy.scope():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', input_shape=(256, 256, 3),padding='same'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),
    
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu',padding='same'),
        tf.keras.layers.MaxPooling2D(2, 2),
    
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu',padding='same'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),
    
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu',padding='same'),
        tf.keras.layers.MaxPooling2D(2, 2),
    
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',padding='same'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.2),
        ###head
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='softmax')
    ]) 
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
model.summary()

#model.compile(optimizer = tf.keras.optimizers.Adam(epsilon=0.01),loss='binary_crossentropy',metrics=['binary_accuracy'])
#model.compile(optimizer = tf.keras.optimizers.Nadam(learning_rate=1e-4),loss='binary_crossentropy',metrics=['binary_accuracy'])
#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# In[4]:


###preparing to training
print("this is images for training")
TRAINING_DIR = "/luidhy_docker/astrodados/DELVE_MORPHOLOGY_DATA/analysis_code_morphology/CNN_testes_simples/default_CNN_smallsample/data_to_work_v5_balance/train/"
train_datagen = ImageDataGenerator(rescale=1.0/255.)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=100,
                                                    class_mode='binary',
                                                    target_size=(256, 256))

print("this is images for validation") 
VALIDATION_DIR = "/luidhy_docker/astrodados/DELVE_MORPHOLOGY_DATA/analysis_code_morphology/CNN_testes_simples/default_CNN_smallsample/data_to_work_v5_balance/val/"
validation_datagen = ImageDataGenerator(rescale=1.0/255.)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=100,
                                                              class_mode='binary',
                                                              target_size=(256, 256))


# In[ ]:


early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
####training the model
history = model.fit_generator(train_generator,
                              epochs=100,
                              verbose=1,
                              callbacks=[early_stopping],
                              validation_data=validation_generator)


# In[ ]:


model.save('results/model_default_smallsample_model4_100epochs.h5')


# In[ ]:


history_frame = pd.DataFrame(history.history)
history_frame.info()
history_frame.to_csv("results/model_default_smallsample_model4_100epochs.csv",index=False)


# In[ ]:


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
plt.savefig('results/Accuracy_model_default_smallsample_model4_100epochs.png',facecolor='white',dpi=200)
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
plt.savefig('results/Loss_model_default_smallsample_model4_100epochs.png',facecolor='white',dpi=200)


# In[ ]:



