# Use a slim Python base image
FROM python:3.12-slim

# Workdir inside the container
WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend ./backend

# Copy frontend index.html
COPY index.html ./index.html

# Copy LCMS assets (icc profiles + transicc binary)
COPY botmcms ./botmcms

# Make sure transicc is executable (won't hurt if it already is)
RUN chmod +x /app/botmcms/icc/transicc || true

# Expose the port Render expects
EXPOSE 10000

# Start FastAPI via uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "10000"]
