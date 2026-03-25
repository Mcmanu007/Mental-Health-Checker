import logging
import pandas as pd
import numpy as np
import yaml
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)

file_handler = logging.FileHandler('model_building.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


#Loading the training data
def load_train_data(file_path:str):
    '''
     Args:
        file_path (str): The path to the training data (e.g., CSV).
        
    Returns:
        pd.DataFrame: A pandas DataFrame containing the data.

    '''
    try:
        data = pd.read_csv(file_path)
        data.dropna(inplace=True)
        logger.debug('Training data loaded successfully')
        return data
    except FileNotFoundError as e:
        logger.error(f'Error in locating the file {file_path}')
    except pd.errors.ParserError as e:
        logger.error(f'Error in parsing the csv file')
    except Exception as e:
        logger.error('An error occured in loading the data')

def convert_word_to_vec(train_data:pd.DataFrame,max_features,ngram_range,max_df) -> tuple:
    '''
     Args:
       train_data: Trainaing data in a pandas Dataframe
    Returns:
       vectors: returns vectorized   
    '''
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=tuple(ngram_range),
            max_df=max_df
        )
        
        X_train = train_data['text'].values
        y_train = train_data['status'].values       

        #performing the tf-idf
        X_train_vec = vectorizer.fit_transform(X_train)
        
        # saving the vectorizer
        joblib.dump(vectorizer,os.path.join(store_pickle(),'vectorizer.pkl'))
        logger.debug('Vecotorization performed successfully')
        return X_train_vec,y_train
    except Exception as e:
        logger.error(f"Error converting the the word to vec {e}")
        raise e


def building_the_model(X_train:np.ndarray,y_train:np.ndarray,max_iter,C,solver):
    '''
    Args:
      X_train: A numpy array for the feature values
      Y_train:  A numpy array for the target values
    Returns:
      model : A trainied logistic regression model
    '''
    try:
        model = LogisticRegression(
            max_iter=max_iter,
            solver=solver,
            C = 1
        )

        model.fit(X_train,y_train)
        logger.debug('Model trained successfully')
        return model
    except Exception as e:
        logger('Failed to trained model')
     

def store_pickle(path:str='./pickle'):
    '''
    Returns:
      path: Creates a path for storing the saved model
    '''
    try:
        os.makedirs(path,exist_ok=True)
        return os.path.abspath(path)
    except Exception as e:
        logger.error('Failure to store the saved model in the specified path')
        raise e
    

def save_model(model):
    try:
        model_path = os.path.join(store_pickle(),'log_model.pkl')
        joblib.dump(model,model_path)
        logger.debug('Saved the trained model')
    except Exception as e:
        logger.error('Error in occured in saving the model')


def model_building():
    try:
        config_path = r'C:\Users\siawc\OneDrive\Desktop\Felix\mental health\config.yaml'

        with open(config_path,'r') as file:
            config = yaml.safe_load(file)

        train_path = config['training_data']['train_data']
        
        #vectorizer parameters
        max_features = config['vectorizer']['max_features']
        ngram_range = config['vectorizer']['ngram_range']
        max_df = config['vectorizer']['max_df']

        # logistic model parameters
        solver = config['logistic']['solver']
        max_iter = config['logistic']['max_iter']
        C = config['logistic']['C']


        train_data = load_train_data(train_path)

        # applying the vectorizer
        X_train,y_train = convert_word_to_vec(train_data,max_features,ngram_range,max_df)
       
       #training the model
        model = building_the_model(
            X_train,
            y_train,
            max_iter,
            C,
            solver
        )

        #saving the model
        save_model(model)
        logger.debug('Model building was successful')
    except Exception as e:
        logger.error('Error in training the model')


if __name__ == '__main__':
    model_building()