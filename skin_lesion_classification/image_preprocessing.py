import numpy as np
import pandas as pd
import os
from glob import glob
from PIL import Image
from keras.utils.np_utils import to_categorical  # used for converting labels to one-hot-encoding
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample


class Preprocessing:
    def __init__(self):
        self.data = pd.read_csv('data/HAM10000_metadata.csv')
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

    def __encode_classes(self):
        le = LabelEncoder()
        le.fit(self.data["dx"])
        LabelEncoder()
        self.data['label'] = le.transform(self.data["dx"])

    def __balance_dataset(self, num_samples=500, shuffle=50, num_classes=7):
        df_balanced = []
        for i in range(num_classes):
            df = self.data[self.data['label'] == i]
            balanced = resample(df, replace=True, n_samples=num_samples, random_state=shuffle)
            df_balanced.append(balanced)
        # Combined back to a single dataframe
        self.data = pd.concat(df_balanced)

    def __read_images(self, image_size=32):
        # To read images based on image ID from the CSV file
        # This is the safest way to read images as it ensures the right image is read for the right ID
        image_path = {}
        list_of_image_paths = glob(os.path.join(r'data', '*', '*.jpg'))

        for path in list_of_image_paths:
            image_path[os.path.splitext(os.path.basename(path))[0]] = path

        # Define the path and add as a new column
        self.data['path'] = self.data['image_id'].map(image_path.get)
        # Use the path to read images.
        self.data['image'] = self.data['path'].map(
            lambda x: np.asarray(Image.open(x).resize((image_size, image_size))))

    def __create_test_train_split(self, split_percentage=0.25, shuffle=50):
        # Convert dataframe column of images into numpy array
        image_column = np.asarray(self.data['image'].tolist())

        # Normalise data (scaling between 0-1)
        image_column = image_column / 255.  # Scale values to 0-1.
        image_labels = self.data['label']  # Assign label values to Y

        # Convert to categorical as this is a multiclass classification problem
        image_categories = to_categorical(image_labels, num_classes=len(list(set(image_labels))))

        # Split to training and testing
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(image_column, image_categories,
                                                                                test_size=split_percentage,
                                                                                random_state=shuffle)

    def prepare_data(self, num_samples=500, shuffle=50, split_percentage=0.25, image_size=32, num_classes=7):
        self.__encode_classes()
        self.__balance_dataset(num_samples, shuffle, num_classes)
        self.__read_images(image_size)
        self.__create_test_train_split(split_percentage, shuffle)
