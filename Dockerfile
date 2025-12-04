# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including LittleCMS tools
RUN apt-get update && apt-get install -y \
    lcms2-utils \
    liblcms2-2 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Create ICC profiles directory
RUN mkdir -p /app/backend/icc_profiles

# Expose port (Render uses PORT env var)
EXPOSE 8000

# Run the application
# Render.com sets the PORT environment variable
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
