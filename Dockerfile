# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Set environment variables
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Default path for mounted Claude projects
ENV CLAUDE_PROJECTS_PATH=/data

# Expose port
EXPOSE 8501

# Run the application
CMD ["uv", "run", "streamlit", "run", "src/app.py"]