import logging
import pandas as pd
import numpy as np
import mlflow
import json
import joblib
import os
import yaml
from sklearn.metrics import accuracy_score,classification_report
from mlflow.models.signature import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer

# logging configuration
logger = logging.getLogger(name='model_evaluation.log')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_evaluation.log')
file_handler.setLevel(logging.DEBUG)

cons_handler = logging.StreamHandler()
cons_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s -%(levelname)s -%(message)s')
file_handler.setFormatter(formatter)
cons_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(cons_handler)


#loading the testing data
def load_test_data(test_path:str):
    try:
        data = pd.read_csv(test_path)
        data.dropna(inplace=True)
        logger.debug('Testing data loaded succesfully')
        return data
    except FileNotFoundError as e:
        logger.error(f'File does not exist {test_path}')
    except pd.errors.ParserError as e:
        logger.error(f'Error in parsing the file {data}')


#loading the validation data
def load_eval_data(eval_path:str):
    try:
        data = pd.read_csv(eval_path)
        data.dropna(inplace=True)
        logger.debug('Validation  data loaded succesfully')
        return data
    except FileNotFoundError as e:
        logger.error(f'Validation dataset does not exist {eval_path}')
    except pd.errors.ParserError as e:
        logger.error(f'Error in parsing the validation data {data}')


# loading the saved model
def load_saved_model(log_model:str):
    try:
        with open(log_model,'rb') as f:
            model = joblib.load(f)
            return model
    except Exception as e:
        logger.error('An error occured in loading the saved model')


def load_vectorizer_model(vec_path:str):
    try:
        with open(vec_path,'rb') as f:
            vector = joblib.load(f)
        logger.debug('Vectorizer loaded from ')
        return vector
    except Exception as e:
        logger.error('Error in loading the pickle vectorizer model')
        raise e


#evaluating the testing data
def evaluate_test_data(X_test:np.ndarray,y_test:np.ndarray,model):
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        report = classification_report(y_test,y_pred,output_dict=True)
        logger.debug('Evaluating the testing data')
        return accuracy, report
    except Exception as e:
        logger.error("Error in evaluating the data")

#evaluating the validation sets
def val_data_predict(X_val:np.ndarray,y_test:np.ndarray,model):
    try:
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_test,y_pred)
        report = classification_report(y_test,y_pred,output_dict=True)
        logger.debug('Successfully evaluated the validation dataset')
        return accuracy, report
    except Exception as e:
        logger.error('Failed to evaluate the validation datasets')

def saving_model_info(model_path:str,run_id:str,file_path:str):
    try:
        model_info = {
            run_id:'run_id',
            model_path:'model_path'
        }
        with open(file_path,'w') as f:
            json.dump(model_info,f,indent=4)
        logger.debug('Model info saved successfully')
    except Exception as e:
        logger.error('Failed to save the model')

def main():
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment('PIPELINES For MENTAL HEALTH CHECK')
    with mlflow.start_run() as run:
        try:
            config_path = r'C:\Users\siawc\OneDrive\Desktop\Felix\mental health\config.yaml'

            with open(config_path,'r') as file:
                config = yaml.safe_load(file)

            test_path = config['testing_data']['test_data']
            val_path = config['validation_data']['eval_data']
            model_path = config['saved_model']['model']
            vec_path = config['vectorizer_model']['vec']

            test_data = load_test_data(test_path) # testing data
            eval_data = load_eval_data(val_path) # validation data
            
            model = load_saved_model(model_path)
            vectorizer = load_vectorizer_model(vec_path)
            
            #prepare the testing data
            X_test = vectorizer.transform(test_data['text'].values)
            y_test = test_data['status'].values

            #preparing the evaluation data
            X_eval = vectorizer.transform(eval_data['text'].values)
            y_eval = eval_data['status'].values


            #create a test dataframe
            tests_example = pd.DataFrame(X_test.toarray()[:10],columns=vectorizer.get_feature_names_out())

            val_example = pd.DataFrame(X_eval.toarray()[:10],columns=vectorizer.get_feature_names_out())

            test_signature = infer_signature(tests_example,model.predict(X_test[:10]))

            val_signature = infer_signature(val_example,model.predict(X_eval[:10]))

            mlflow.sklearn.log_model(
                model,
                'LogisticRegressionModel',
                signature=test_signature,
                input_example=tests_example
            )
            model_path = 'Logistic Regression Model'
            saving_model_info(
                run.info.run_id, 
                model_path,
                'model_info.json'
            )

            whitelist = ['vectorizer', 'logistic']
            #logging the parameters of the vectorizer and logistic regression model
            params_to_log = {}
            for section in whitelist:
                if section in config:
                    for key, value in config[section].items():
                        params_to_log[f"{section}_{key}"] = value
            mlflow.log_params(params_to_log)
            
            accuracy,report = evaluate_test_data(X_test,y_test,model)

            for label,metrics in report.items():
                if isinstance(metrics,dict):
                    for metric,value in metrics.items():
                        mlflow.log_metric(f"test{label}_{metric}",value)

            print(f'Accuracy {accuracy}')
        except Exception as e:
            logger.error(f'An error occured in validating the data{e}')


if __name__ == '__main__':
    main()


                      

