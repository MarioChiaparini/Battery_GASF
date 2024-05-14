from scipy import io
from pyts.image import GramianAngularField
import os
import datetime
import seaborn as sns
from google.colab import drive
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
from skimage.io import imread
import matplotlib.image as mpimg

def load_data_from_all_files(data_dir):
    mat_files = glob.glob(os.path.join(data_dir, '*.mat'))
    all_datasets = []

    for mat_file in mat_files:
        battery = os.path.basename(mat_file).split('.')[0]
        mat = loadmat(mat_file)
        print(f'Total data in dataset {battery}: ', len(mat[battery][0, 0]['cycle'][0]))
        counter = 0
        last_entry_per_cycle = {}

        for i in range(len(mat[battery][0, 0]['cycle'][0])):
            row = mat[battery][0, 0]['cycle'][0, i]
            if row['type'][0] == 'discharge':
                ambient_temperature = row['ambient_temperature'][0][0]
                date_time = datetime.datetime(int(row['time'][0][0]),
                                              int(row['time'][0][1]),
                                              int(row['time'][0][2]),
                                              int(row['time'][0][3]),
                                              int(row['time'][0][4])) + datetime.timedelta(seconds=int(row['time'][0][5]))
                data = row['data']
                capacity = data[0][0]['Capacity'][0][0]
                for j in range(len(data[0][0]['Voltage_measured'][0])):
                    voltage_measured = data[0][0]['Voltage_measured'][0][j]
                    current_measured = data[0][0]['Current_measured'][0][j]
                    temperature_measured = data[0][0]['Temperature_measured'][0][j]
                    current_load = data[0][0]['Current_load'][0][j]
                    voltage_load = data[0][0]['Voltage_load'][0][j]
                    time = data[0][0]['Time'][0][j]
                    last_entry_per_cycle[counter + 1] = [counter + 1, ambient_temperature, date_time, capacity,
                                                         voltage_measured, current_measured, temperature_measured,
                                                         current_load, voltage_load, time, battery]
                counter += 1

        print(list(last_entry_per_cycle.values())[0])
        all_datasets.extend(last_entry_per_cycle.values())

    return pd.DataFrame(data=all_datasets,
                        columns=['cycle', 'ambient_temperature', 'datetime',
                                 'capacity', 'voltage_measured',
                                 'current_measured', 'temperature_measured',
                                 'current_load', 'voltage_load', 'time', 'battery'])

def calculate_soh(df, initial_capacity=2.0):
    df['SoH'] = (df['capacity'] / initial_capacity) * 100
    return df

def categorize_soh(soh):
  if soh > 80:
    return 'Good'

  elif 50 <= soh <= 80:
    return 'Moderate'
  else:
    return 'Poor'
  
def series_to_gaf(series, method):
  min_value = series.min()
  max_value = series.max()
  scaled_series = (2 * (series - min_value) / (max_value - min_value)) - 1
  gaf = GramianAngularField(method=f"{method}", image_size=len(scaled_series.reshape))
  data = gaf.transform(scaled_series.reshape(1,-1))
  return data[0]


def crete_gaf_db(base_path, labels, features, rescale_size):
  
  X_gaf = []
  y_labels = []

  for label in labels:
    dir =os.path.join(base_path, label)
    for feature in features:
      imag_file = os.path.join(dir, f"{feature}.png")
      if os.path.exists(imag_file):
        image = imread(imag_file, as_gray=True)
        imag_resized = resize(image, (rescale_size, rescale_size), anti_aliasing=True)
        X_gaf.append(imag_resized)
        y_labels.append(feature)
  X_gaf = np.array(X_gaf, dtype='float32')
  y_labels = np.array(y_labels)

  X_gaf = np.expand_dims(X_gaf, axis= -1)
  return X_gaf, y_labels


data_dir = '/content/drive/MyDrive/Notes/Data/ARC1'
dataset = load_data_from_all_files(data_dir)

df_final = calculate_soh(dataset)
B0006 = df_final[df_final['battery'] == "B0006"]

sns.set_style("darkgrid")
plt.figure(figsize=(12, 8))
plt.plot(B0006['cycle'], B0006['SoH'])
plt.xlabel('cycle')
plt.title('SoH - B0006')

df_final['HealthCategory'] = df_final['SoH'].apply(categorize_soh)
B0006['HealthCategory'] = df_final['SoH'].apply(categorize_soh)
dataframe = df_final.groupby([df_final['cycle'], 'battery'])[['SoH']].first().unstack()




LABEL_NAMES = ['Good', 'Moderate']
features = [
     'voltage_measured','current_measured', 
     'temperature_measured', 'current_load', 
     'voltage_load'
]

label_column = 'HealthCategory'
labels = B0006[label_column].unique()

img_sz = 50
method = 'summation'
gaf = GramianAngularField(image_size=img_sz, method=method)

base_path_summation = "/content/drive/MyDrive/Notes/GAF/"

for label in labels:
    label_df = B0006[B0006[label_column] == label].copy()
    
    label_dir = os.path.join(base_path_summation, label)
    os.makedirs(label_dir, exist_ok=True)
    for feature in features:
        feature_data = label_df[feature]
        
        if feature_data.isna().all():
            print(f"All values are NaN for feature '{feature}' in label '{label}'. Skipping.")
            continue
        
        feature_data.fillna(feature_data.mean(), inplace=True)
        if feature_data.nunique() <= 1:
            print(f"Feature '{feature}' for label '{label}' is constant. Skipping.")
            continue
        min_val = feature_data.min()
        max_val = feature_data.max()
        normalized_data = (2 * (feature_data - min_val) / (max_val - min_val)) - 1 
        gaf_img = gaf.fit_transform(normalized_data.values.reshape(1, -1))[0]
        img_save_path = os.path.join(label_dir, f"{feature}.png")
        plt.imsave(img_save_path, gaf_img, cmap="hot")


for label in LABEL_NAMES:
    # Create a new figure and axes for each charge policy
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(20, 20))
    fig.suptitle(f"Gramian Angular Field: Charge Policy '{label}'", fontsize=20)
    # Directory for the current charge policy
    policy_dir = os.path.join(base_path_summation, label)
    # Loop through each feature and plot the GAF image
    for i, feature in enumerate(features):
        # Define the path for the GAF image file
        img_file = os.path.join(policy_dir, f"{feature}.png")
        if os.path.exists(img_file):
            img = mpimg.imread(img_file)
            ax = axs[i // 2, i % 2]  
            ax.imshow(img)
            ax.set_title(f"{feature}", fontsize=10)
            ax.axis('off')
        else:
            print(f"No image found for {feature} in charge policy {label}.")
    plt.show()