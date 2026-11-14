
FROM python:3.11-slim

WORKDIR /srv

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt gunicorn huggingface_hub

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["sh", "-c", "hf download adyashaexe/GastricSentinel-models --repo-type model --local-dir /srv && exec gunicorn -w 1 --threads 4 -b 0.0.0.0:5000 'app.app:create_app()'"]
