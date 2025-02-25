FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY scripts/run_client.py /app/run_client.py

ENTRYPOINT ["python", "run_client.py"]