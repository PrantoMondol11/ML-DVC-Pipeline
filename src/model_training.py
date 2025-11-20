import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml


log_dir="logs"
os.makedirs(log_dir,exist_ok=True)
logger=logging.getLogger("model_training")
logger.setLevel("DEBUG")

file_log_path=os.path.join(log_dir,"model_traing.log")
file_handler=logging.FileHandler(file_log_path)
file_handler.setLevel("DEBUG")


console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

formattor=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formattor)
file_handler.setFormatter(formattor)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(file_path:str):
    try:
        with open(file_path,'r') as file:
            params=yaml.safe_load(file)
            logger.debug("Parameter retrived from %s",file_path)
            return params
    except FileNotFoundError as e:
        logger.error("File not found : %s",e)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error:%s",e)
        raise

def load_data(file_path):
    try:
        df=pd.read_csv(file_path)
        df.fillna('',inplace=True)
        logger.debug("Data Loaded and NaNs filled from %s",file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to pharse the csv file : %s",e)
        raise
    except Exception as e :
        logger.error("Unexpected Error occure while loading the data")
        raise
    
def train_model(x_train:np.ndarray,y_train:np.ndarray,params:dict) -> RandomForestClassifier:
    
    try:
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("The nuber of sample in x_train and y_train must be the same.")
        logger.debug("Initializing RandomForest model With parameters: %s",params)
        clf=RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])
        
        logger.debug('Model training started with %d samples',x_train.shape[0])
        clf.fit(x_train,y_train)
        logger.debug("Model Training completed")
        return clf
    except ValueError as e:
        logger.error("ValueError During model Training: %s",e)
        raise
    
def save_model(model,file_path:str)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
    except FileNotFoundError as e:
        logger.error("File path not found: %s",e)
        raise
    except Exception as e:
        logger.error("Error occure during saving the model: %s",e)
        raise
    
def main():
    try:
       
        params=load_params('params.yaml')['model_training']
        train_data=load_data('./data/processed/train_tfid.csv')
        x_train=train_data.iloc[:,:-1].values
        y_train=train_data.iloc[:,-1].values
    
        clf=train_model(x_train,y_train,params)
        
        model_save_path='models/model.pkl'
        save_model(clf,model_save_path)
    except Exception as e:
        logger.error("Failed to complete the model building: %s",e)
        raise
if __name__=='__main__':
    main()            
    
    