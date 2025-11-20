import os 
import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml
log_dir="logs"
os.makedirs(log_dir,exist_ok=True)
logger=logging.getLogger("feature_engineering.log")
logger.setLevel("DEBUG")

file_log_path=os.path.join(log_dir,"feature_engineering.log")
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
    
def applytfid(train_data,test_data,max_feature):
    try:
        vectorizer=TfidfVectorizer(max_features=max_feature)
        
        x_train=train_data['text'].values
        y_train=train_data['target'].values
        x_test=test_data['text'].values
        y_test=test_data['target'].values
        
        x_train_bow=vectorizer.fit_transform(x_train)
        x_test_bow=vectorizer.fit_transform(x_test)
        
        train_df=pd.DataFrame(x_train_bow.toarray())
        train_df['label']=y_train
        
        test_df=pd.DataFrame(x_test_bow.toarray())
        test_df['label']=y_test
        
        logger.debug("Bag of data applied and Transformed")
        return  train_df,test_df
    except Exception as e:
        logger.error("Error During bag of Words Transformation : %s ",e)
        raise
def save_data(df,file_path):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index=True)
        logger.debug("Data saved to %s",file_path)
    except Exception as e :
        logger.error("Unexpected error occure during saving the data: %s",e)
        raise

def main():
    try:
        
        params=load_params(file_path='params.yaml')
        max_feature=params['feature_engineering']['max_feature']
        
        train_data=load_data('./data/interim/train_processed.csv')
        test_data=load_data("./data/interim/test_processed.csv")
        
        train_df,test_df=applytfid(train_data,test_data,max_feature)
        
        save_data(train_df,os.path.join("./data","processed","train_tfid.csv"))
        save_data(test_df,os.path.join("./data","processed","test_tfid.csv"))
    except Exception as e:
        logger.error("FAiled to complete the feature engineering Process: %s",e)
        print(f"Error;{e}")
        
if __name__=="__main__":
    main()