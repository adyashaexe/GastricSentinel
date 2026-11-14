FROM python:3.11-slim
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn
COPY . .
ENV PYTHONUNBUFFERED=1
EXPOSE 5000
# one worker: Grad-CAM hooks share the model; scale with more containers instead
CMD ["gunicorn", "-w", "1", "--threads", "4", "-b", "0.0.0.0:5000", "app.app:create_app()"]
