FROM python:3.11-slim

WORKDIR /app
COPY . /app

# System deps: ffmpeg for video work, ca-certs for HTTPS
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# Railway sets PORT; JSON CMD won't expand env vars, so use sh -c
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
