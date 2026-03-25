import logging
import pandas as pd
import os
import yaml
import re
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder


#logging configuration for data preprocessing category
logger = logging.getLogger(name='data_preprocessing')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('data_preprocessing.log')
file_handler.setLevel(logging.DEBUG)

cons_handler = logging.StreamHandler()
cons_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
cons_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(cons_handler)



#loading the data
def load_data(data_path:str):
    """
    Args:
        file_path (str): The path to the data file (e.g., CSV).
        
    Returns:
        pd.DataFrame: A pandas DataFrame containing the d
    """
    try:
        data = pd.read_csv(data_path)
        data.dropna(inplace=True)
        data.drop('Unique_ID',axis=True)
        logger.debug('Data loaded succesfully')
        encoder = LabelEncoder()
        data['status'] = encoder.fit_transform(data['status'])
        return data
    except FileNotFoundError as e:
        logger.error(f'file not found{data_path}')
        raise e
    except pd.errors.ParserError as e:
        logger.error('Error in parsing the csv file')
        raise e
    except Exception as e:
        logger.error('Error in loading the data')
        raise e
    

def preprocesses_text(text):
    """
    Returns:
        cleaned_text :A cleaned and preprocessed text
    """
    text = text.lower()
    
    text = re.sub(r'[^a-z\s]', '', text)
    
    tokens = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def normalize_text(data:pd.DataFrame):
    """
    Args:
        Takes In the unprocessed text
    Returns:
        preprocessed_csv: It returns a ready to use data for building
    """
    try:
        
        data['text'] = data['text'].apply(preprocesses_text)
        
        logger.debug("Tweets normalized successfully")
        return data
    except Exception as e:
        logger.error('Failed to normalize the tweets')


def save_preprocess_data(train:pd.DataFrame,test:pd.DataFrame,path:str='.data/preprocessed data'):
    """
     Args:
       train: Takes in the train data as a pandas Dataframe
       test: Takes in the test data as a pandas Dataframe
       path: Path in saving the data
     Returns:
       data: A cleaned csv file
    """
    try:
        os.makedirs(path,exist_ok=True)
        train.to_csv(os.path.join(path,'train_prepocessed.csv'),index=False)
        test_path = os.path.join(path,'test_preprocessed.csv')
        val_path = os.path.join(path, 'eval_preprocessed.csv')

        val_data,test_data= split_data(test, test_size=0.2, random_state=42)
        test_data.to_csv(test_path, index=False)
        val_data.to_csv(val_path, index=False)
    except Exception as e:
        logger.error('An error occured in saving the data')
        raise e
    

def split_data(data:pd.DataFrame, test_size:float, random_state:int)-> pd.DataFrame:
    ''''
    splitting the data into evaluation and testing datasets
    ''' 
    try:
        eval_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
        return eval_data, test_data
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise e
    
    
def preprocessing_stage():
    try:
        config_path = r'C:\Users\siawc\OneDrive\Desktop\Felix\mental health\config.yaml'

        with open(config_path,'r') as f:
            config = yaml.safe_load(f)

        train_data_path = config['preprocessing_data']['train_path']
        test_data_path =  config['preprocessing_data']['test_path']

        train_data = load_data(train_data_path)
        test_data = load_data(test_data_path)

        train_preproccessed = normalize_text(train_data)
        test_preprocessed = normalize_text(test_data)

        save_preprocess_data(train_preproccessed,test_preprocessed)
    except Exception as e:
        logger.error('An error occured in the preprocessing stage')
        raise e
    
if __name__ == '__main__':
    preprocessing_stage()

    



       
