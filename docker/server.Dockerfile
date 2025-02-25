FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY scripts/run_server.py /app/run_server.py

RUN mkdir -p /app/models

EXPOSE 8080

ENTRYPOINT ["python", "run_server.py"]