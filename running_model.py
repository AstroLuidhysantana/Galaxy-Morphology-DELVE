import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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
from tensorflow.keras.preprocessing import image
from keras.optimizers import Adam
###to run on multigpu
if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
    strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
else:
    strategy = tf.distribute.get_strategy()
    
    
tf.test.is_built_with_cuda()
tf.config.list_physical_devices('GPU')


###load the model
model = keras.models.load_model('results/model_default_smallsample_model4.h5')

####load folder the images that needs classification
folder = '/luidhy_docker/astrodados/DELVE_MORPHOLOGY_DATA/analysis_code_morphology/CNN_testes_simples/image_rgb_norm/'
#folder = '/luidhy_docker/astrodados/DELVE_MORPHOLOGY_DATA/analysis_code_morphology/CNN_testes_simples/default_CNN_smallsample/data_to_work_v5_balance/test/all_test/'
####TRYING TO SAVE RUN THE MODEL WITH THE PREDICT PROBS 
images_names = []
spiral_probabilities = []
elliptical_probabilities = []
predicted_classes = []
for filename in os.listdir(folder):
    img_path = os.path.join(folder, filename)
    img = image.load_img(img_path, target_size=(256, 256))
    img = image.img_to_array(img) / 255.
    img = np.expand_dims(img, axis=0)
    
    ####oredictions using the model
    probabilities = model.predict(img)
    spiral_prob = probabilities[0, 1]
    elliptical_prob = probabilities[0, 0]
    
    ###predicted class based on the highest probability
    predicted_class = "spiral" if spiral_prob > elliptical_prob else "eliptical"
    
    ###storring the results
    images_names.append(os.path.splitext(filename)[0])
    spiral_probabilities.append(spiral_prob)
    elliptical_probabilities.append(elliptical_prob)
    predicted_classes.append(predicted_class)
    
results_df = pd.DataFrame({
    'id': images_names,
    'Spiral_Probability': spiral_probabilities,
    'Elliptical_Probability': elliptical_probabilities,
    'Predicted_Class': predicted_classes
})

results_df.to_csv('results/results_alltest_classification_largebestsample_model4.csv', index=False)        
