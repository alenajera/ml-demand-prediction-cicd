FROM python:3.10-slim
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY model /app/model

EXPOSE 5000
CMD ["python", "app/app.py"]