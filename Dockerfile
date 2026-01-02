FROM python:3.11-slim

# Install FFmpeg and dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Verify FFmpeg
RUN ffmpeg -version && ffmpeg -encoders | grep libx264

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler and overlay files
COPY handler.py .
COPY overlays/ ./overlays/

CMD ["python", "-u", "handler.py"]
