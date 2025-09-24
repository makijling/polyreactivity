FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.hf_cache

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git wget hmmer && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash appuser
WORKDIR /workspace

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chown -R appuser:appuser /workspace
USER appuser

CMD ["python", "-m", "polyreact.predict", "--help"]
