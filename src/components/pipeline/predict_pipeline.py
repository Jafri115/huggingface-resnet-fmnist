import os
import sys
import pandas as pd
from src.exceptions import CustomException
from src.logger import logging
from src.utils import load_object
import cv2
import random
import numpy as np
class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            data_scaled = preprocessor.transform(features)
            predictions = model.predict(data_scaled)
            return predictions
        except Exception as e:
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                    path: str
                    ):
        
        self.path = path
        

        
    def get_data_as_frame(self):
        try:
            data = {
                "path": [self.path],

            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def preprocess_image(self,image):
        image_resized = cv2.resize(image, (224, 224))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_augmented = cv2.flip(image,random.randint(-1, 1)) 
        return image_augmented


        