FROM python:3.11-slim

WORKDIR /app

COPY services/model/requirements.txt /tmp/model-requirements.txt
RUN pip install --no-cache-dir -r /tmp/model-requirements.txt

COPY scripts/retrain.py scripts/promotion.py ./scripts/
COPY services/model/preprocess.py ./services/model/preprocess.py
COPY services/model/models/ ./services/model/models/
COPY data/ ./data/

CMD ["python", "scripts/retrain.py", "--min-feedback", "200"]
