FROM python:3.13.12-bookworm

WORKDIR /app

COPY  api .

RUN pip install -r requirements.txt

COPY pickle/*.pkl ./pickle/ 

COPY config.yaml .
EXPOSE 8000




CMD ["uvicorn", "app:app", "--host", "0.0.0", "--port", "8000"]
