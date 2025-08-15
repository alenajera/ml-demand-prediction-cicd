# 1. Imagen base oficial de Python
FROM python:3.10-slim

# 2. Directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiar los archivos necesarios
# Copiamos requirements.txt primero para aprovechar la cache
COPY app/requirements.txt .

# 4. Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar el resto del proyecto
COPY app/ ./app/
COPY model/ ./model/

# 6. Puerto expuesto para Flask
EXPOSE 5000

# 7. Comando para iniciar la app en producción con Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app.app:app"]