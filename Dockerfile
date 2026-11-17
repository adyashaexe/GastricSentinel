FROM python:3.11-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
# one process: the model is loaded once and Grad-CAM runs under an asyncio lock; scale with more containers
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
