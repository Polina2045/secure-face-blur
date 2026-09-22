from python:3.11-slim

WORKDIR /app

# Системные библиотеки для OpenCV (даже headless-версии нужны glib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости отдельно, чтобы использовать кеш слоёв при сборке
COPY requirements.txt .
# PyTorch ставим отдельно из CPU-индекса (без CUDA -> образ меньше и легче)
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces по умолчанию слушает порт 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]