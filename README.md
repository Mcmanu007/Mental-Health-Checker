# Mental Health Check

This project aims to build and machine learning models for predicting the state of my mind of an individual by analyzing the messages of the individual. The workflow includes data preprocessing, exploratory data analysis (EDA), model training, evaluation, and API deploymen

## Project Structure

```
Mental Health/
|── data/
│      ├── preropcessed data/
│          ├── eval_preprocessed.csv
|          ├── test_preprocessed.csv
|          ├── train_preprocessed.csv
│      └── raw/
│           ├── test.csv
│           ├── train.csv
├── api/
│   ├── app.py           # API for model inference
│   └── base.py          # Base utilities for API
├── eda&experimentation/
│   ├── eda.ipynb        # Exploratory Data Analysis
│   ├── baseline_experimentation.ipynb
│   ├── logistic_experiment.ipynb
│   
├── src/
│   ├── data_modelling/
│   │   ├── data_ingestion.py
│   │   └── data_preprocess.py
│   └── model/
│       ├── model_building.py
│       └── model_evaluation.py
├── mlartifacts/         # Model artifacts and requirements
├── pkl/                 # Pickled models
├── requirements.txt     # Python dependencies
├── conda.yaml           # Conda environment file
├── python_env.yaml      # Python environment config
├── dvc.yaml             # DVC pipeline config
├── config.yaml          # Project configuration
└── model_info.json      # Model metadata
```

## Features
- Data ingestion and preprocessing
- Exploratory data analysis (EDA)
- Multiple ML models (logistic regression, random forest, etc.)
- Model evaluation and comparison
- Model tracking with MLflow
- API for model inference
- Reproducible environments (conda, requirements.txt)
- DVC for data and model versioning

## Getting Started

### 1. Clone the repository
```bash
git clone <repo-url>
cd cardio
```

### 2. Set up the environment
Using Conda:
```bash
conda env create -f conda.yaml
conda activate cardio
```
Or using pip:
```bash
pip install -r requirements.txt
```

### 3. Data Preparation
- Place your raw data in the appropriate folder (see `eda&experimentation/data/`).
- Use the notebooks in `eda&experimentation/` for EDA and initial modeling.

### 4. Model Training & Evaluation
- Run scripts in `src/data_modelling/` and `src/model/` for data processing and model training.
- Track experiments and models with MLflow (see `mlruns/`).

### 5. API Deployment
- Start the API with:
```bash
python API/app.py
```
- The API will serve predictions using the trained model.

### 6. DVC Usage
- Use DVC to manage data and model versioning:
```bash
dvc repro
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](LICENSE)

## Acknowledgements
- [MLflow](https://mlflow.org/)
- [DVC](https://dvc.org/)
- [Scikit-learn](https://scikit-learn.org/)
