# Étape de build
FROM python:3.12-slim AS builder

WORKDIR /app

COPY . /app

RUN python -m venv /opt/venv
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Étape finale
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY . /app

EXPOSE 8080 

ENV FLASK_RUN_HOST=0.0.0.0
ENV PORT 8080  
ENV PATH="/opt/venv/bin:$PATH"

CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"] 