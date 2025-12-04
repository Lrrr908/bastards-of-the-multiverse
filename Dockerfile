FROM python:3.11.9-slim-bookworm

# Install LittleCMS runtime and utilities (gives you transicc)
RUN apt-get update && apt-get install -y --no-install-recommends \
      liblcms2-2 \
      lcms2-utils \
    && rm -rf /var/lib/apt/lists/*

# Set workdir inside the container
WORKDIR /app

# Copy the repo into the container
COPY . .

# Install Python dependencies for the backend
RUN pip install --no-cache-dir -r backend/requirements.txt

# Render will map 10000 from inside container to outside
EXPOSE 10000

# Start FastAPI with uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "10000"]
