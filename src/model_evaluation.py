import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
import logging


log_dir="logs"
os.makedirs(log_dir,exist_ok=True)
logger=logging.getLogger("model_evaluation")
logger.setLevel("DEBUG")

file_log_path=os.path.join(log_dir,"model_evaluation.log")
file_handler=logging.FileHandler(file_log_path)
file_handler.setLevel("DEBUG")


console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

formattor=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formattor)
file_handler.setFormatter(formattor)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model(file_path:str):
    try:
        with open(file_path,'rb') as file:
            model = pickle.load(file)
        logger.debug("model loaded from %s",file_path)
        return model
    except  FileNotFoundError as e:
        logger.error("File not found in: %s ",e)
        raise
    except Exception as e:
        logger.error("Unexpected error occured while loading the data: %s",e)
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
        logger.error("Unexpected Error occure while loading the data: %s",e)
        raise
    
    
def model_eval(clf,x_test:np.ndarray,y_test:np.ndarray)->dict:
    try:
        y_pred= clf.predict(x_test)
        y_pred_proba=clf.predict_proba(x_test)[:,1]
        
        accuracy=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall= recall_score(y_test,y_pred)
        auc=roc_auc_score(y_test,y_pred_proba)
        
        metrics_dict={
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'auc':auc
            
        }
        logger.debug("Model evaluation metrics calculated")
        return metrics_dict
    except Exception as e:
        logger.error("Error during model Evaluation:%s",e)
        raise
    
def save_metrics(metrics:dict,file_path:str)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'w') as file:
            json.dump(metrics,file,indent=4)
        logger.debug("Metrics Saved to %s",file_path)
    except Exception as e:
        logger.error("Error occure while saving the metrics:%s",e)
        raise
    
def main():
    try:
        clf=load_model('./models/model.pkl')
        test_data=load_data('./data/processed/test_tfid.csv')
        x_test=test_data.iloc[:,:-1].values
        y_test=test_data.iloc[:,-1].values
        
        metrics=model_eval(clf,x_test,y_test)
        
        save_metrics(metrics,'reports/metrics.json')
    except Exception as e:
        logger.error("Failed to complete the data evaluation process: %s",e)
        print(f"Error:{e}")
    
if __name__=='__main__':
        main()