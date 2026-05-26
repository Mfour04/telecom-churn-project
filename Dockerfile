FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install .

CMD ["python", "src/models/train.py"]