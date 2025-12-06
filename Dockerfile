# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Install uv for faster package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for building packages and Docling
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Configure uv to use system Python (required for Docker)
ENV UV_SYSTEM_PYTHON=1

# Copy the requirements file into the container
COPY requirements-docker.txt ./requirements.txt

# Install any needed packages specified in requirements.txt using uv (much faster than pip)
RUN uv pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
COPY . .

# Make entrypoint script executable
RUN chmod +x docker-entrypoint.sh

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run entrypoint script which disables tracing then starts the API
ENTRYPOINT ["./docker-entrypoint.sh"]
