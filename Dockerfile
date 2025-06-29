# Use a more specific Python 3.11 slim image, aligning with your local environment
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first to leverage Docker cache
# This layer will only be rebuilt if requirements.txt changes
COPY requirements.txt .

# Install dependencies with --no-cache-dir for smaller image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code into the container
# NOTE: This copies files from your build context. Use .dockerignore to exclude
# large or unnecessary files (like .git, venv/, mlruns/, data/).
# For development with docker-compose volumes, this COPY is less critical for source code
# (as the volume mount overwrites it), but it's good practice for general image builds.
COPY . .

# Expose the port your FastAPI application listens on
EXPOSE 8000

# Command to run the application using uvicorn
# --host 0.0.0.0 is crucial for allowing external connections to the container
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]