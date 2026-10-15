# Production Dockerfile compatible with local Docker & Hugging Face Spaces
FROM python:3.11-slim

# Prevent Python from writing bytecode and enable immediate stdout flushing
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DOCKER_CONTAINER=true \
    HOST=0.0.0.0 \
    PORT=7860 \
    HOME=/home/user

# Hugging Face Spaces requires non-root user with UID 1000
RUN useradd -m -u 1000 user

WORKDIR /home/user/app

# Install dependencies in a separate layer for maximum Docker cache reuse
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application assets, HTML frontend, backend server, dataset, and report
COPY --chown=user:user app.py .
COPY --chown=user:user app.html .
COPY --chown=user:user medical_insurance.csv .
COPY --chown=user:user report.pdf .

# Pre-train the model during image build so the container boots instantly
RUN python -c "import app; app.get_or_train_model()" && chown user:user best_model.joblib

# Switch to non-root user
USER user

# Hugging Face Spaces expects traffic on port 7860
EXPOSE 7860

# Container healthcheck querying the model telemetry endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/api/model_info')" || exit 1

# Launch the MediRisk server
CMD ["python", "app.py"]
