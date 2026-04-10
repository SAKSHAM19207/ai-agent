# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies including Node.js (needed to build React frontend)
RUN apt-get update && apt-get install -y \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast python package installer)
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh

# Set the working directory in the container
WORKDIR /app

# Copy the entire project directory into the container
COPY . /app/

# --- Build the Frontend React Application ---
WORKDIR /app/frontend
# Install dependencies
RUN npm install
# Produce the optimized static build
RUN npm run build

# --- Build the Python Backend ---
WORKDIR /app

# Install all backend dependencies systematically using uv (into the system Python)
RUN uv pip install --system fastapi uvicorn pypdf python-multipart python-dotenv huggingface_hub httpx
# Install the root OpenEnv/JAOE package requirements
RUN uv pip install --system -e .

# Hugging Face Spaces strictly requires applications to bind to port 7860
EXPOSE 7860

# Command to run the application using uvicorn on port 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
