import discord
from discord.ext import commands
import numpy as np
from google.cloud import speech_v1
from pydub import AudioSegment
import json
import logging
import webrtcvad
import asyncio

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TranscriberBot')

class Config:
    def __init__(self):
        self.load_config()

    def load_config(self):
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.token = config['token']
                self.prefix = config.get('prefix', '!')
                self.language = config.get('language', 'es-ES')
                self.vad_aggressiveness = config.get('vad_aggressiveness', 3)
                self.silence_duration = config.get('silence_duration', 900)  # ms
                self.sample_rate = config.get('sample_rate', 16000)
                logger.info('Configuración cargada exitosamente')
        except Exception as e:
            logger.error(f'Error al cargar la configuración: {e}')
            raise

class AudioProcessor:
    def __init__(self, config, speech_client):
        self.vad = webrtcvad.Vad(config.vad_aggressiveness)
        self.sample_rate = config.sample_rate
        self.frame_duration = 30  # ms
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        self.silence_frames = 0
        self.max_silence_frames = config.silence_duration // self.frame_duration
        self.speech_client = speech_client
        self.audio_buffer = []

    def process_audio(self, audio_chunk):
        try:
            is_speech = self.vad.is_speech(audio_chunk, self.sample_rate)
            if is_speech:
                self.audio_buffer.append(audio_chunk)
                self.silence_frames = 0
            else:
                self.silence_frames += 1
                if self.silence_frames > self.max_silence_frames and self.audio_buffer:
                    self.transcribe_audio()
                    self.audio_buffer = []
                    self.silence_frames = 0
        except Exception as e:
            logger.error(f'Error procesando audio: {e}')

    def transcribe_audio(self):
        audio_data = b''.join(self.audio_buffer)
        audio_segment = AudioSegment(
            data=audio_data,
            sample_width=2,
            frame_rate=self.sample_rate,
            channels=1
        )
        audio_bytes = audio_segment.raw_data

        audio = speech_v1.RecognitionAudio(content=audio_bytes)
        config = speech_v1.RecognitionConfig(
            encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate,
            language_code='es-ES'
        )

        response = self.speech_client.recognize(config=config, audio=audio)
        for result in response.results:
            logger.info(f'Transcripción: {result.alternatives[0].transcript}')

class MySink(discord.sinks.Sink):
    def __init__(self, bot, ctx):
        super().__init__()
        self.bot = bot
        self.ctx = ctx
        self.processor = AudioProcessor(bot.config, bot.speech_client)

    def write(self, data, user):
        self.processor.process_audio(data)

    async def cleanup(self):
        if self.processor.audio_buffer:
            self.processor.transcribe_audio()

class VoiceTranscriberBot(commands.Bot):
    def __init__(self, config):
        intents = discord.Intents.default()
        intents.voice_states = True
        intents.message_content = True
        super().__init__(command_prefix=config.prefix, intents=intents)
        
        self.config = config
        self.speech_client = speech_v1.SpeechClient()
        self.recording_channels = set()

    async def setup_hook(self):
        await self.add_cog(TranscriberCog(self))
        logger.info('Bot inicializado y listo')

class TranscriberCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def start(self, ctx):
        try:
            if ctx.author.voice:
                channel = ctx.author.voice.channel
                if ctx.voice_client is None:
                    await channel.connect()
                else:
                    await ctx.voice_client.move_to(channel)
                await ctx.send('✅ ¡Conectado y listo para transcribir!')
                logger.info(f'Bot unido al canal de voz en {ctx.guild.name}')
            else:
                await ctx.send('❌ Necesitas estar en un canal de voz.')
                return

            if ctx.guild.id in self.bot.recording_channels:
                await ctx.send('⚠️ ¡Ya estoy grabando!')
                logger.warning('Bot ya está grabando en este servidor')
                return

            self.bot.recording_channels.add(ctx.guild.id)
            await ctx.send('🎙️ ¡Comenzando a transcribir! Detectaré automáticamente cuando alguien hable.')
            logger.info(f'Iniciando grabación en {ctx.guild.name}')

            sink = MySink(self.bot, ctx)
            ctx.voice_client.start_recording(sink, self.on_recording_done, ctx)

        except Exception as e:
            logger.error(f'Error al iniciar la grabación: {e}')
            await ctx.send('❌ Error al iniciar la grabación')

    async def on_recording_done(self, sink, ctx):
        await sink.cleanup()
        self.bot.recording_channels.remove(ctx.guild.id)
        await ctx.send('🛑 Grabación detenida.')
        logger.info(f'Grabación detenida en {ctx.guild.name}')

    @commands.command()
    async def stop(self, ctx):
        try:
            if ctx.guild.id not in self.bot.recording_channels:
                await ctx.send('❌ No estoy grabando en este servidor.')
                return

            ctx.voice_client.stop_recording()

        except Exception as e:
            logger.error(f'Error al detener la grabación: {e}')
            await ctx.send('❌ Error al detener la grabación')

if __name__ == "__main__":
    config = Config()
    bot = VoiceTranscriberBot(config)
    bot.run(config.token)