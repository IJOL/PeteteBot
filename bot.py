import discord
import json
import logging
from google.cloud import speech_v1
from pydub import AudioSegment
import webrtcvad
import asyncio


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TranscriberBot')

# Configuration loading
class Config:
    def __init__(self):
        self.load_config()

    def load_config(self):
        try:
            with open('config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.token = config['token']
                self.language = config.get('language', 'es-ES')
                self.vad_aggressiveness = config.get('vad_aggressiveness', 3)
                self.silence_duration = config.get('silence_duration', 900)  # ms
                self.sample_rate = config.get('sample_rate', 16000)
                logger.info('Configuration loaded successfully')
        except Exception as e:
            logger.error(f'Error loading configuration: {e}')
            raise

# Transcriber bot
class TranscriberBot:
    def __init__(self, config):
        self.config = config
        self.speech_client = speech_v1.SpeechClient()
        self.audio_buffer = []
        self.channel = None

    def set_channel(self, channel):
        self.channel = channel

    def transcribe_audio(self, audio_data):
        audio = speech_v1.RecognitionAudio(content=audio_data)
        config = speech_v1.RecognitionConfig(
            encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.config.sample_rate,
            language_code=self.config.language
        )

        response = self.speech_client.recognize(config=config, audio=audio)
        for result in response.results:
            transcription = result.alternatives[0].transcript
            logger.info(f'Transcription: {transcription}')
            if self.channel:
                asyncio.run_coroutine_threadsafe(self.channel.send(transcription), asyncio.get_event_loop())

# Audio processor
class AudioProcessor:
    def __init__(self, config, transcriber_bot):
        self.vad = webrtcvad.Vad(config.vad_aggressiveness)
        self.sample_rate = config.sample_rate
        self.frame_duration = 30  # ms
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        self.silence_frames = 0
        self.max_silence_frames = config.silence_duration // self.frame_duration
        self.transcriber_bot = transcriber_bot

    def process_audio(self, audio_chunk):
        self.transcriber_bot.audio_buffer.append(audio_chunk)
        total_audio_length = len(b''.join(self.transcriber_bot.audio_buffer))

        if total_audio_length < self.frame_size:
            return

        is_speech = self.vad.is_speech(audio_chunk[:self.frame_size], self.sample_rate)

        if is_speech:
            self.silence_frames = 0
        else:
            self.silence_frames += 1
            if self.silence_frames > self.max_silence_frames and self.transcriber_bot.audio_buffer:
                self.transcribe_audio()
                self.transcriber_bot.audio_buffer = []
                self.silence_frames = 0

    def transcribe_audio(self):
        audio_data = b''.join(self.transcriber_bot.audio_buffer)
        audio_segment = AudioSegment(
            data=audio_data,
            sample_width=2,
            frame_rate=self.sample_rate,
            channels=1
        )
        audio_bytes = audio_segment.raw_data
        self.transcriber_bot.transcribe_audio(audio_bytes)

# Custom sink for real-time processing
class MySink(discord.sinks.Sink):
    def __init__(self, config, transcriber_bot):
        super().__init__()
        self.processor = AudioProcessor(config, transcriber_bot)

    def write(self, data, user):
        self.processor.process_audio(data)

    async def cleanup(self):
        if self.processor.transcriber_bot.audio_buffer:
            self.processor.transcribe_audio()

# Discord bot setup
bot = discord.Bot()
connections = {}
config = Config()
transcriber_bot = TranscriberBot(config)

async def finished_callback(sink, channel: discord.TextChannel, *args):
    recorded_users = [f"<@{user_id}>" for user_id, audio in sink.audio_data.items()]
    await sink.vc.disconnect()
    await channel.send(f"Finished! Recorded audio for {', '.join(recorded_users)}.")

@bot.event
async def on_ready():
    print(f"{bot.user} is ready and online!")

@bot.slash_command(name="start", description="Start recording")
async def start(ctx: discord.ApplicationContext):
    voice = ctx.author.voice

    if not voice:
        return await ctx.respond("You're not in a vc right now")

    vc = await voice.channel.connect()
    connections.update({ctx.guild.id: vc})
    transcriber_bot.set_channel(ctx.channel)

    vc.start_recording(
        MySink(config, transcriber_bot),
        finished_callback,
        ctx.channel,
    )

    await ctx.respond("The recording has started!")

@bot.slash_command(name="stop", description="Stop recording")
async def stop(ctx: discord.ApplicationContext):
    if ctx.guild.id in connections:
        vc = connections[ctx.guild.id]
        vc.stop_recording()
        del connections[ctx.guild.id]
        await ctx.delete()
    else:
        await ctx.respond("Not recording in this guild.")

bot.run(config.token)