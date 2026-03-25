
# Project Memory - Mental Health MLOps

## Project Structure
- `src/`: Contains source code for data modelling and model training.
- `.data/`: Data storage (managed by DVC).
- `config.yaml`: Centralized configuration for paths and hyperparameters.
- `dvc.yaml`: DVC pipeline definition.

## Conventions
- **Path Management**: Always use relative paths from the project root in Python scripts.
- **Dependencies**: New dependencies must be added to `requirements.txt`.
- **NLTK**: Scripts using NLTK should include `nltk.download()` calls for required datasets to ensure compatibility with CI environments.
- **MLflow**: Tracking URI is conditional on environment (local vs CI).

## CI/CD
- GitHub Actions workflow is defined in `.github/workflows/mlops-ci-workflow.yaml`.
- Uses DVC for data versioning and pipeline execution (`dvc repro`).
- DVC pull and repro require appropriate remote credentials (e.g., `GDRIVE_CREDENTIALS_DATA` for Google Drive) configured in GitHub Secrets.
- Always include `nltk.download()` calls for required datasets in preprocessing scripts to ensure compatibility with clean CI environments.
