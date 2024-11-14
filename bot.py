import discord
from discord.sinks import Sink
import asyncio
import json
import logging
from google.cloud import speech_v1
from google.cloud import translate_v2
from google.oauth2 import service_account
from typing import Dict, Optional
import numpy as np
import scipy.signal
import threading
import queue

class Config:
    def __init__(self, config_path='config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        logging.basicConfig(
            level=self.config.get('logging', {}).get('level', 'INFO'),
            format=self.config.get('logging', {}).get('format', '%(asctime)s - %(levelname)s - %(message)s'),
            handlers=[logging.StreamHandler()]
        )

    def get(self, path, default=None):
        current = self.config
        for part in path.split('.'):
            if isinstance(current, dict):
                current = current.get(part, default)
            else:
                return default
        return current

class AudioBuffer:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.active = True
        self.logger = logging.getLogger(f'AudioBuffer-{user_id}')
        self.audio_queue = queue.Queue()
        self.logger.info(f"Creado nuevo buffer de audio para usuario {user_id}")

    def add_audio(self, data: bytes) -> None:
        self.logger.debug(f"Añadiendo audio de tamaño {len(data)}")
        resampled_data = self.resample_audio(data)
        self.audio_queue.put(resampled_data)

    def get_audio(self) -> bytes:
        self.logger.debug("Obteniendo audio de la cola")
        return self.audio_queue.get()

    def resample_audio(self, audio: bytes, original_rate: int = 48000, target_rate: int = 16000) -> bytes:
        self.logger.debug(f"Resampleando audio de {original_rate}Hz a {target_rate}Hz")
        audio_np = np.frombuffer(audio, dtype=np.int16)
        resampled_audio = scipy.signal.resample_poly(audio_np, target_rate, original_rate)
        return resampled_audio.astype(np.int16).tobytes()

class VoiceTranscriptionSink(Sink):
    def __init__(self, bot):
        super().__init__()
        self.bot = bot
        self.text_channel = None
        self.logger = logging.getLogger('VoiceTranscriptionSink')
        self.user_buffers: Dict[int, AudioBuffer] = {}
        self.processing_threads: Dict[int, threading.Thread] = {}
        self.user_clients: Dict[int, speech_v1.SpeechClient] = {}
        
        credentials = service_account.Credentials.from_service_account_file(
            bot.config.get('google.credentials_path'),
            scopes=[
                'https://www.googleapis.com/auth/cloud-platform',
                'https://www.googleapis.com/auth/speech-to-text',
                'https://www.googleapis.com/auth/cloud-translation'
            ]
        )
        
        self.credentials = credentials
        languages = bot.config.get('speech.languages', ['es-ES'])
        self.streaming_config = speech_v1.StreamingRecognitionConfig(
            config=speech_v1.RecognitionConfig(
                encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=languages[0],
                alternative_language_codes=languages[1:]
            ),
            interim_results=True
        )

    def write(self, data: bytes, user_id: int) -> None:
        self.logger.debug(f"Escribiendo datos de audio para el usuario {user_id}")
        if user_id not in self.user_buffers:
            self.logger.info(f"Iniciando nuevo procesamiento de audio para usuario {user_id}")
            self.user_buffers[user_id] = AudioBuffer(user_id)
            self.user_clients[user_id] = speech_v1.SpeechClient(credentials=self.credentials)
            self.processing_threads[user_id] = threading.Thread(
                target=self._process_audio, args=(user_id,)
            )
            self.processing_threads[user_id].start()
            self.logger.info(f"Hilo de procesamiento iniciado para usuario {user_id}")
        
        self.user_buffers[user_id].add_audio(data)

    def _process_audio(self, user_id: int) -> None:
        buffer = self.user_buffers[user_id]
        self.logger.info(f"Iniciando bucle de procesamiento para usuario {user_id}")
        
        while buffer.active:
            self.logger.info(f"Esperando datos de audio para el usuario {user_id}")
            audio_data = buffer.get_audio()
            
            try:
                self.logger.info(f"Procesando audio para el usuario {user_id}")
                requests = [
                    speech_v1.StreamingRecognizeRequest(audio_content=audio_data)
                ]
                
                responses = self.user_clients[user_id].streaming_recognize(
                    self.streaming_config,
                    requests
                )
                self.logger.info(f"Respuestas recibidas para el usuario {user_id}")
                for response in responses:
                    self.logger.info(f"{response}")
                    for result in response.results:
#                        if result.is_final:
                        transcript = result.alternatives[0].transcript
                        self.logger.info(f"Transcripción final para el usuario {user_id}: {transcript}")
                        self._send_transcription(user_id, transcript)
                        
            except Exception as e:
                self.logger.error(f"Error procesando audio para usuario {user_id}: {e}")

        self.logger.info(f"Finalizando bucle de procesamiento para usuario {user_id}")

    def _send_transcription(self, user_id: int, text: str) -> None:
        self.logger.debug(f"Enviando transcripción para el usuario {user_id}")
        if self.text_channel:
            user = self.bot.get_user(user_id)
            username = user.name if user else f"Usuario {user_id}"
            asyncio.run_coroutine_threadsafe(
                self.text_channel.send(f"{username}: {text}"), self.bot.loop
            )
        else:
            self.logger.error("No hay canal de texto configurado")

    def cleanup(self) -> None:
        self.logger.info("Iniciando limpieza de recursos del sink")
        user_ids = list(self.user_buffers.keys())  # Hacer una copia de las claves
        for user_id in user_ids:
            self.user_buffers[user_id].active = False
            if user_id in self.processing_threads:
                self.processing_threads[user_id].join()
        
        self.user_buffers.clear()
        self.processing_threads.clear()
        self.user_clients.clear()
        self.logger.info("Recursos del sink limpiados correctamente")

class VoiceBot(discord.Bot):
    def __init__(self, config_path: str):
        super().__init__()
        self.config = Config(config_path)
        self.logger = logging.getLogger('VoiceBot')

    async def setup_hook(self) -> None:
        self.add_listener(self.on_voice_state_update)

    async def on_voice_state_update(
        self, 
        member: discord.Member, 
        before: discord.VoiceState, 
        after: discord.VoiceState
    ) -> None:
        if before.channel != after.channel:
            if after.channel and after.channel.guild.voice_client:
                self.logger.info(f"Usuario {member.name} (ID: {member.id}) se unió al canal")
                
            elif before.channel and before.channel.guild.voice_client:
                self.logger.info(f"Usuario {member.name} (ID: {member.id}) dejó el canal")
                if isinstance(before.channel.guild.voice_client.sink, VoiceTranscriptionSink):
                    sink = before.channel.guild.voice_client.sink
                    if member.id in sink.user_buffers:
                        sink.user_buffers[member.id].active = False
                        sink.user_buffers[member.id].ready.set()
                        sink.processing_tasks[member.id].cancel()
                        del sink.user_buffers[member.id]
                        del sink.processing_tasks[member.id]

def setup_commands(bot):
    @bot.slash_command()
    async def join(ctx: discord.ApplicationContext):
        """Une el bot al canal de voz"""
        if not ctx.author.voice:
            await ctx.respond("¡Necesitas estar en un canal de voz!")
            return
            
        channel = ctx.author.voice.channel
        if ctx.voice_client:
            await ctx.voice_client.disconnect()
            
        await channel.connect()
        
        sink = VoiceTranscriptionSink(bot)
        sink.text_channel = ctx.channel
        
        def on_recording_finished(sink, *args):
            sink.cleanup()
        
        ctx.voice_client.start_recording(sink, on_recording_finished)
        
        await ctx.respond(f"Conectado a {channel.name}")

    @bot.slash_command()
    async def leave(ctx: discord.ApplicationContext):
        """Desconecta el bot del canal de voz"""
        if not ctx.voice_client:
            await ctx.respond("¡No estoy conectado a ningún canal!")
            return

        # Detener grabación y limpiar recursos
        if isinstance(ctx.voice_client.sink, VoiceTranscriptionSink):
            ctx.voice_client.stop_recording()
            ctx.voice_client.sink.cleanup()

        # Desconectar del canal
        await ctx.voice_client.disconnect()
        await ctx.respond("¡Desconectado del canal de voz!")        

def main():
    try:
        bot = VoiceBot('config.json')
        setup_commands(bot)
        bot.run(bot.config.get('bot.token'))
    except Exception as e:
        logging.error(f"Error ejecutando el bot: {e}")

if __name__ == "__main__":
    main()