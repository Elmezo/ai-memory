FROM python:3.11-slim

# Prevent Python from writing .pyc and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies if needed (e.g. curl for health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Run as an unprivileged, non-root user (defense in depth).
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
ENV HOME=/home/appuser
USER appuser

# Expose REST API and MCP ports
EXPOSE 8000 8001

# Health check against REST API
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default startup command (runs REST API)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
