FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PORT=8000 OPENBELL_STORAGE_BACKEND=supabase OPENBELL_COLLECTORS_ENABLED=0
ENV TZ=Asia/Kolkata
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 tzdata && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-runtime.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-runtime.txt
COPY . .
EXPOSE 8000
# One worker owns the in-process collectors. Multiple workers duplicate polling.
CMD ["sh", "-c", "exec python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
