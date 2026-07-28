FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt requirements-wandb.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements-wandb.txt

COPY train_gpt2.py fineweb.py hellaswag.py terminal_loger.py ./
COPY configs ./configs

ENTRYPOINT ["python", "train_gpt2.py"]
