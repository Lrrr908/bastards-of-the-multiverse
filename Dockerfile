# Starter Dockerfile for BotMCMS on Render

FROM python:3.12-slim

# Install LittleCMS runtime on the server
RUN apt-get update && apt-get install -y --no-install-recommends \
    liblcms2-2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy backend and install Python dependencies
COPY backend ./backend
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy static site
COPY index.html ./index.html

EXPOSE 10000

# Start FastAPI
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "10000"]
