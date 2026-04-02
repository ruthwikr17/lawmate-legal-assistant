# Use Python slim for a lightweight container
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for sentence-transformers, HNSW, PyTorch if necessary)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file from the root directory level
# (Assuming the Docker build context is the root directory or requirements are moved in)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY backend/ ./backend/

# IMPORTANT: Copy the built ChromaDB folder into the Docker image!
# This ensures HuggingFace Spaces boots instantly with pre-loaded vectors
# without losing data on ephemeral restarts.
COPY db/ ./db/

# Optional: Add any static scripts/models if necessary
# COPY storage/ ./storage/

# Expose port (HF Spaces defaults to 7860, Render uses PORT env var)
# We read the OS PORT in main.py directly.
EXPOSE 7860
EXPOSE 8000

# Start the FastAPI server using the backend module
CMD ["python", "backend/main.py"]
