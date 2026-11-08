# Образ с PyTorch: работает и на CPU, и на GPU (запуск с --gpus all)
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY doe/ doe/
COPY weights/ weights/
COPY run.py train.py ./
COPY examples/ examples/

# Входные комплекты монтируются в /data/input, результаты пишутся в /data/output
ENTRYPOINT ["python", "run.py"]
CMD ["--input", "/data/input", "--output", "/data/output"]
