FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (including OpenGL for OpenCV)
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libx11-6 \
    libxkbcommon0 \
    libdbus-1-3 \
    libfontconfig1 \
    libharfbuzz0b \
    libfreetype6 \
    libjpeg62-turbo \
    libpng16-16 \
    libtiff6 \
    libwebp7 \
    libopenjp2-7 \
    liblcms2-2 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Run the Flask app
CMD ["python", "app.py"]
