FROM denoland/deno:bin AS deno

# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=deno /deno /usr/local/bin/deno

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg ca-certificates opencv-data git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch 1.3.1 \
    https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git \
    /opt/bgutil-ytdlp-pot-provider \
    && cd /opt/bgutil-ytdlp-pot-provider/server \
    && deno install --allow-scripts=npm:canvas --frozen

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . /app

# Railway provides PORT. Docker JSON CMD doesn't expand env vars,
# so use a shell form command.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
 
