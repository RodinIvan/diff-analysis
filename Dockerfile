FROM python:3.12.7

WORKDIR /app
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "/app/main.py"]
