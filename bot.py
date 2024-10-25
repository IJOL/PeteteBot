import discord
from discord.ext import commands
import asyncio
import webrtcvad
import numpy as np
from google.cloud import speech_v1
from pydub import AudioSegment
import json
import logging

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
    def __init__(self, config):
        self.vad = webrtcvad.Vad(config.vad_aggressiveness)
        self.sample_rate = config.sample_rate
        self.frame_duration = 30  # ms
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        self.silence_frames = 0
        self.max_silence_frames = config.silence_duration // self.frame_duration

    def process_audio(self, audio_chunk):
        try:
            is_speech = self.vad.is_speech(audio_chunk, self.sample_rate)
            return is_speech
        except Exception as e:
            logger.error(f'Error procesando audio: {e}')
            return False

class VoiceTranscriberBot(commands.Bot):
    def __init__(self, config):
        intents = discord.Intents.default()
        intents.voice_states = True
        intents.message_content = True
        super().__init__(command_prefix=config.prefix, intents=intents)
        
        self.config = config
        self.speech_client = speech_v1.SpeechClient()
        self.audio_processors = {}
        self.recording_channels = set()

    async def setup_hook(self):
        await self.add_cog(TranscriberCog(self))
        logger.info('Bot inicializado y listo')

class AudioSegmentHandler:
    def __init__(self, bot, channel):
        self.bot = bot
        self.channel = channel
        self.current_segment = []
        self.processor = AudioProcessor(bot.config)
        
    async def process_audio_buffer(self, audio_data):
        if len(audio_data) == 0:
            return

        try:
            audio_segment = AudioSegment(
                data=b''.join(audio_data),
                sample_width=2,
                frame_rate=48000,
                channels=2
            )
            
            audio_segment = audio_segment.set_channels(1).set_frame_rate(self.bot.config.sample_rate)
            await self.transcribe_segment(audio_segment)
        except Exception as e:
            logger.error(f'Error procesando buffer de audio: {e}')
            
    async def transcribe_segment(self, audio_segment):
        try:
            audio_content = audio_segment.raw_data
            audio = speech_v1.RecognitionAudio(content=audio_content)
            
            config = speech_v1.RecognitionConfig(
                encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.bot.config.sample_rate,
                language_code=self.bot.config.language,
                enable_automatic_punctuation=True,
            )
            
            response = self.bot.speech_client.recognize(config=config, audio=audio)
            
            for result in response.results:
                transcript = result.alternatives[0].transcript
                if transcript.strip():
                    await self.channel.send(f'Transcripción: {transcript}')
                    logger.info(f'Transcripción exitosa: {transcript[:50]}...')
        
        except Exception as e:
            logger.error(f'Error en la transcripción: {e}')
            await self.channel.send('❌ Error al procesar el audio')

class TranscriberCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.recording = {}
        self.segment_handlers = {}

    @commands.command()
    async def join(self, ctx):
        try:
            if ctx.author.voice:
                channel = ctx.author.voice.channel
                if ctx.voice_client is None:
                    await channel.connect()
                else:
                    await ctx.voice_client.move_to(channel)
                self.segment_handlers[ctx.guild.id] = AudioSegmentHandler(self.bot, ctx.channel)
                await ctx.send('✅ ¡Conectado y listo para transcribir!')
                logger.info(f'Bot unido al canal de voz en {ctx.guild.name}')
            else:
                await ctx.send('❌ Necesitas estar en un canal de voz.')
        except Exception as e:
            logger.error(f'Error al unirse al canal: {e}')
            await ctx.send('❌ Error al unirse al canal de voz')

    @commands.command()
    async def start(self, ctx):
        try:
            if not ctx.voice_client:
                await ctx.send('❌ ¡Necesito estar en un canal de voz!')
                return

            if ctx.guild.id in self.recording:
                await ctx.send('⚠️ ¡Ya estoy grabando!')
                return

            self.recording[ctx.guild.id] = True
            await ctx.send('🎙️ ¡Comenzando a transcribir! Detectaré automáticamente cuando alguien hable.')
            logger.info(f'Iniciando grabación en {ctx.guild.name}')

            handler = self.segment_handlers[ctx.guild.id]
            
            def audio_receiver(user, audio_data):
                if handler.processor.process_audio(audio_data):
                    handler.current_segment.append(audio_data)
                    handler.processor.silence_frames = 0
                else:
                    handler.processor.silence_frames += 1
                    
                    if (handler.processor.silence_frames >= handler.processor.max_silence_frames 
                        and handler.current_segment):
                        asyncio.create_task(handler.process_audio_buffer(handler.current_segment))
                        handler.current_segment = []

            ctx.voice_client.listen(audio_receiver)

        except Exception as e:
            logger.error(f'Error al iniciar la grabación: {e}')
            await ctx.send('❌ Error al iniciar la grabación')

    @commands.command()
    async def stop(self, ctx):
        try:
            if ctx.guild.id not in self.recording:
                await ctx.send('⚠️ ¡No estoy grabando!')
                return

            handler = self.segment_handlers[ctx.guild.id]
            if handler.current_segment:
                await handler.process_audio_buffer(handler.current_segment)
            
            ctx.voice_client.stop_listening()
            del self.recording[ctx.guild.id]
            await ctx.send('🛑 ¡Transcripción detenida!')
            logger.info(f'Grabación detenida en {ctx.guild.name}')

        except Exception as e:
            logger.error(f'Error al detener la grabación: {e}')
            await ctx.send('❌ Error al detener la grabación')

    @commands.command()
    async def leave(self, ctx):
        try:
            if ctx.voice_client:
                if ctx.guild.id in self.recording:
                    ctx.voice_client.stop_listening()
                    del self.recording[ctx.guild.id]
                await ctx.voice_client.disconnect()
                await ctx.send('👋 ¡Desconectado del canal de voz!')
                logger.info(f'Bot desconectado en {ctx.guild.name}')
            else:
                await ctx.send('⚠️ No estoy conectado a ningún canal de voz.')

        except Exception as e:
            logger.error(f'Error al desconectar: {e}')
            await ctx.send('❌ Error al desconectar')

def main():
    try:
        config = Config()
        bot = VoiceTranscriberBot(config)
        bot.run(config.token)
    except Exception as e:
        logger.critical(f'Error crítico al iniciar el bot: {e}')

if __name__ == "__main__":
    main()