diff --git a/Dockerfile b/Dockerfile
index f6dd94c8a02353d790ecc00f56f562b4228dc7a2..85fc385a7d77577905082a4eab322c24d34e71f3 100644
--- a/Dockerfile
+++ b/Dockerfile
@@ -1,14 +1,20 @@
+# syntax=docker/dockerfile:1
 FROM python:3.11-slim
 
+ENV PYTHONDONTWRITEBYTECODE=1 \
+    PYTHONUNBUFFERED=1
+
 WORKDIR /app
-COPY . /app
 
 RUN apt-get update && apt-get install -y --no-install-recommends \
     ffmpeg ca-certificates \
     && rm -rf /var/lib/apt/lists/*
 
-RUN pip install --no-cache-dir -r requirements.txt
+COPY requirements.txt /app/requirements.txt
+RUN python -m pip install --no-cache-dir -r requirements.txt
+
+COPY . /app
 
 # Railway provides PORT. Docker JSON CMD doesn't expand env vars,
 # so use a shell form command.
 CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
