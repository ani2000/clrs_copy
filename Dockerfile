FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV CKPT_DIR=/app/checkpoints_v3
ENV SERVER_HOST=0.0.0.0

RUN chmod +x start.sh

# in-browser model bundles (128-hidden weights from checkpoints_v3) are generated at build time
RUN python model_export.py models

EXPOSE 8766 8767

CMD ["./start.sh"]
