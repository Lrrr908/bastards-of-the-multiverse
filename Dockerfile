# Simple, stable base image
FROM python:3.12-slim

# Workdir inside container
WORKDIR /app

# Install Python deps
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend and frontend
COPY backend ./backend
COPY index.html ./index.html
COPY botmcms ./botmcms

EXPOSE 10000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "10000"]
