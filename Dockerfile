FROM python:3.11.9-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
      liblcms2-2 \
      liblcms2-dev \
      gcc \
    && rm -rf /var/lib/apt/lists/*

# Set workdir inside the container
WORKDIR /app

# Copy the repo into the container
COPY . .

# Install Python dependencies for the backend
RUN pip install --no-cache-dir -r backend/requirements.txt

# Install Python LCMS2 library
RUN pip install --no-cache-dir pillow-simd || pip install --no-cache-dir pillow

# Render will map 10000 from inside container to outside
EXPOSE 10000

# Start FastAPI with uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "10000"]
