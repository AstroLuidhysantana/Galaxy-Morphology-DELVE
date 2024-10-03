import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing import image
print("all imports fine")

# Set environment variables and GPU configuration
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# Enable memory growth for GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#setup the strategy to use all the gpus available
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1","/gpu:2","/gpu:3","/gpu:4","/gpu:5","/gpu:6","/gpu:7"])

# Model and images aths
model_path = '/luidhy_docker/astrodados/DELVE_MORPHOLOGY_DATA/CNN_models/KFOLD_binary_model/model_results/complex_model_50epochs.h5'
#folder = '/luidhy_docker/astrodados/DELVE_MORPHOLOGY_DATA/CNN_models/training_selection/original_data_60k_towork/test/spheroid/'
#folder = '/luidhy_docker/astrodados/DELVE_MORPHOLOGY_DATA/DELVE_MORPHOLOGY_ALLSTAMPS_v2/16_split/images_png/'
folder = '/luidhy_docker/DELVEIMAGENS/morphology_images/7_split/images_png/'
#folder = '/luidhy_docker/astrodados/DELVE_MORPHOLOGY_DATA/DELVE_MORPHOLOGY_ALLSTAMPS_v2/18_split/images_png/'
# Load the model in the gpus
print(folder)
with strategy.scope():
    model = tf.keras.models.load_model(model_path)

#function to load and apply the normalization to the images
def load_and_preprocess_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [256, 256])
    img = tf.cast(img, tf.float32) / 255.0
    return img

#create the dataset list
list_ds = tf.data.Dataset.list_files(os.path.join(folder, '*.png'))

# Map the loading and preprocessing function to the dataset
image_ds = list_ds.map(lambda x: load_and_preprocess_image(x))

# choose the batch size
batch_size = 512
image_ds = image_ds.batch(batch_size)
image_ds = image_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Initialize lists to store results
images_names = []
disk_probabilities = []
spheroid_probabilities = []
predicted_classes = []

#Run the predictions using the complex model
for batch in image_ds:
    probabilities = model.predict(batch)
    
    # Extract the file names and predictions for each image in the batch
    batch_filenames = list_ds.take(batch_size).as_numpy_iterator()
    for i, prob in enumerate(probabilities):
        disk_prob = prob[0]
        spheroid_prob = prob[1]
        predicted_class = "disk" if disk_prob > spheroid_prob else "spheroid"
        
        
        filename = os.path.basename(next(batch_filenames)).decode('utf-8')
        images_names.append(os.path.splitext(filename)[0])
        disk_probabilities.append(disk_prob)
        spheroid_probabilities.append(spheroid_prob)
        predicted_classes.append(predicted_class)

#saving the results
print("Saving the results")
results_df = pd.DataFrame({
    'QUICK_OBJECT_ID': images_names,
    'Disk_Probability': disk_probabilities,
    'Spheroid_Probability': spheroid_probabilities,
    'Predicted_Class': predicted_classes
})

# Save the results to a CSV file
results_df.to_csv('morphology_classification_7_split.csv', index=False)

print("Inference completed and results saved.")
