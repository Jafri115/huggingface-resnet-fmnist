from dataclasses import dataclass
import numpy as np
import os
import sys
import tensorflow as tf

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    '''
    DataTransformationConfig: A class for holding the configuration for data transformation
    '''
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        This function will return the data transformer object for image preprocessing
        '''
        try:
            # For Fashion MNIST, we'll create a simple preprocessing function
            # that normalizes pixel values and reshapes the images
            
            def preprocess_images(images):
                """
                Normalize and reshape images
                Args:
                    images: numpy array of shape (num_samples, 28, 28)
                Returns:
                    processed_images: numpy array of shape (num_samples, 28, 28, 1)
                """
                # Normalize pixel values to [0, 1]
                images = images.astype('float32') / 255.0
                # Add channel dimension
                images = np.expand_dims(images, -1)
                return images
            
            logging.info("Created Fashion MNIST image preprocessor")
            return preprocess_images
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path: str, test_path: str, train_labels_path: str, test_labels_path: str):
        logging.info("Data Transformation Started for Fashion MNIST")
        try:
            # Load numpy arrays
            train_images = np.load(train_path)
            test_images = np.load(test_path)
            train_labels = np.load(train_labels_path)
            test_labels = np.load(test_labels_path)
            
            logging.info("Loaded train and test images and labels")
            
            # Get preprocessing function
            preprocessor_fn = self.get_data_transformer_object()
            
            # Preprocess images
            train_images = preprocessor_fn(train_images)
            test_images = preprocessor_fn(test_images)
            
            # One-hot encode labels
            num_classes = 10
            train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
            test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)
            
            logging.info("Preprocessed images and encoded labels")
            
            # Save the preprocessor function
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_fn
            )
            
            return (
                (train_images, train_labels),
                (test_images, test_labels),
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            logging.error("Error occurred during data transformation")
            raise CustomException(e, sys)