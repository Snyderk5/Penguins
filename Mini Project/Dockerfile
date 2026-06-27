FROM python:3.11-slim

WORKDIR /app

ENV MPLBACKEND=Agg

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "code/clean_eda.py"]