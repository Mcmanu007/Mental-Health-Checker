import logging
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import yaml

# logging configuration
logger = logging.getLogger(name='data_ingestion')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('data_ingestion.log')
file_handler.setLevel(logging.DEBUG)

cons_handler = logging.StreamHandler()
cons_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s-%(levelname)s-%(message)s')
file_handler.setFormatter(formatter)
cons_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(cons_handler)


# loading the data
def load_data(data:str):
    """
    Args:
        file_path (str): The path to the data file (e.g., CSV).
        
    Returns:
        pd.DataFrame: A pandas DataFrame containing the data.
    """
    try:
        logger.info('Loading the csv file')
        data_path = pd.read_csv(data)
        logger.debug('Data loaded successfully')
        return data_path
        
    except FileNotFoundError as e:
        logger.error(f"The file can't be found {data}")
        raise e
    except pd.errors.ParserError:
        logger.error(f'An error occured in parsinf the csv file')
        raise e
    except Exception as e:
        logger.error('An error occured in loading the data')
        raise e


# splitting the data
def split_data(data:pd.DataFrame,test_size: int,random_state:int):
    """
    Args:
       data: The pandas data frame
       random_state: state of reproducibility
       test_size: dividing the data

    Returns:
        Train and Test DataFrame
    """
    try:
        train,test = train_test_split(data, test_size=test_size, random_state=random_state)
        logger.info('Splitting the data was succesful')
        return train,test
    except Exception as e:
        logger.error('Error in splitting the data')
        raise e
    

def saving_data(train:pd.DataFrame,test:pd.DataFrame,path:str='.data/raw'):
    """
    Args:
       train: Takes in the train data as a pandas Dataframe
       test: Takes in the test data as a pandas Dataframe
       path: Path in saving the data
    """
    try:
        os.makedirs(path,exist_ok=True)
        train.to_csv(os.path.join(path, 'train.csv'),index=False)
        test.to_csv(os.path.join(path,'test.csv'),index=False)
        logger.debug('Saving the data was successful')
    except Exception as e:
        logger.error('An error occurred in saving the data')
        raise e
    

def ingest_data_stage():
    """
    Initializes the ingest data stage

    Args:
       Configuration path = Path to yaml configuration file
    """
    try:
        config_path = 'config.yaml'

        with open(config_path,'r') as file:
            config = yaml.safe_load(file)

        data_path = config['data']['data_path']
        test_size = config['data']['test_size']
        random_state = config['data']['random_state']
        
        data = load_data(data_path)
        train,test = split_data(data,test_size=test_size,random_state=random_state)
        saving_data(train,test)
    except Exception as e:
        logger.error('An error occured in the ingest data stage')
        raise e
    

if __name__ == '__main__':
    ingest_data_stage()

