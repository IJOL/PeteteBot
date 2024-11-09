# Dockerfile
FROM python:3.11-slim

# Instalar dependencias del sistema necesarias para webrtcvad y audio
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    ffmpeg \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Copiar archivos de requerimientos primero (mejor caching)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código del bot
COPY . .
# Crear un usuario no root para ejecutar el bot
RUN useradd -m botuser
RUN chown -R botuser:botuser /app
RUN mkdir /app/temp_audio && chown botuser:botuser /app/temp_audio

USER botuser


# Comando para ejecutar el bot
CMD ["python", "bot.py"]
