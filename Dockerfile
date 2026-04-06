# Use Python slim for a lightweight container
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for sentence-transformers, HNSW, OpenCV, PyTorch)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file from the root directory level
# (Assuming the Docker build context is the root directory or requirements are moved in)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY backend/ ./backend/

# IMPORTANT: The local database is 1.9 GB, which is too large for GitHub/Hugging Face
# without Git LFS. We will let ChromaDB initialize a fresh, empty vector store on boot.
RUN mkdir -p db/
# COPY db/ ./db/

# Optional: Add any static scripts/models if necessary
# COPY storage/ ./storage/

# Expose port (HF Spaces defaults to 7860)
EXPOSE 7860
ENV PORT=7860

# Start the FastAPI server using the backend module
CMD ["python", "backend/main.py"]
