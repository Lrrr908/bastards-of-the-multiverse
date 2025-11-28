# Use a slim Python base image
FROM python:3.12-slim

# Install system deps needed for LittleCMS + transicc
RUN apt-get update && apt-get install -y --no-install-recommends \
      liblcms2-2 \
      lcms2-utils \
    && rm -rf /var/lib/apt/lists/*

# Workdir inside the container
WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend ./backend

# Copy frontend index.html
COPY index.html ./index.html

# Copy LCMS assets (ICC profiles etc). We no longer rely on the
# custom transicc binary here, but keeping the folder is fine.
COPY botmcms ./botmcms

# Expose the port Render expects
EXPOSE 10000

# Start FastAPI via uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "10000"]
