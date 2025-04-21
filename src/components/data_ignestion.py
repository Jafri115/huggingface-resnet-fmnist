import os
import sys
import numpy as np
import tensorflow as tf
from src.exceptions import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    '''
    DataIngestionConfig: A class for holding the configuration for data ingestion
    '''
    train_data_path: str = os.path.join("artifacts", "train.npy")
    test_data_path: str = os.path.join("artifacts", "test.npy")
    raw_data_path: str = os.path.join("artifacts", "raw.npy")
    train_labels_path: str = os.path.join("artifacts", "train_labels.npy")
    test_labels_path: str = os.path.join("artifacts", "test_labels.npy")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")
        try:
            # Load Fashion MNIST dataset
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
            logging.info("Loaded Fashion MNIST dataset")
            
            # Create artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Combine all data for potential future use
            x_full = np.concatenate((x_train, x_test))
            y_full = np.concatenate((y_train, y_test))
            
            # Save raw data
            np.save(self.ingestion_config.raw_data_path, x_full)
            
            # Save processed splits
            np.save(self.ingestion_config.train_data_path, x_train)
            np.save(self.ingestion_config.test_data_path, x_test)
            np.save(self.ingestion_config.train_labels_path, y_train)
            np.save(self.ingestion_config.test_labels_path, y_test)
            
            logging.info("Saved train/test splits to artifacts directory")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.train_labels_path,
                self.ingestion_config.test_labels_path
            )
        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path, train_labels_path, test_labels_path = data_ingestion.initiate_data_ingestion()
        
        data_transformation = DataTransformation()
        train_array, test_array, _ = data_transformation.initiate_data_transformation(
            train_data_path, 
            test_data_path,
            train_labels_path,
            test_labels_path
        )
        
        model_trainer = ModelTrainer()
        best_score = model_trainer.initiate_model_trainer(train_array, test_array)
        print(f"Best Model Score: {best_score}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise CustomException(e, sys)