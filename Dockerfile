# Usar imagen base de Python 3.9
FROM python:3.9-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de requirements primero para aprovechar cache de Docker
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar todos los archivos del proyecto
COPY . .

# Exponer puerto para Streamlit
EXPOSE 8501

# Comando por defecto para mantener el contenedor corriendo
CMD ["python", "-c", "import time; print('🐳 Docker listo - todas las dependencias instaladas'); time.sleep(999999)"]