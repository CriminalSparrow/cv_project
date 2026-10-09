ARG TRITON_IMAGE=nvcr.io/nvidia/tritonserver:23.12-py3
FROM ${TRITON_IMAGE}
ENV PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends wkhtmltopdf \
    && rm -rf /var/lib/apt/lists/*
ENV QT_QPA_PLATFORM=offscreen
COPY . .
EXPOSE 8000 8001 8002
HEALTHCHECK --interval=10s --timeout=5s --start-period=90s --retries=6 \
  CMD curl -fsS http://localhost:8000/v2/health/ready || exit 1
CMD ["tritonserver", "--model-repository=/workspace/model_repository", "--strict-readiness=true", "--disable-auto-complete-config", "--allow-gpu-metrics=false", "--pinned-memory-pool-byte-size=0"]
