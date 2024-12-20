import discord
import webrtcvad
import asyncio
import wave
import os
import json
import logging
from google.cloud import speech_v1
from google.cloud import translate_v2
from google.oauth2 import service_account
from discord.sinks import Sink
import numpy as np
from pathlib import Path
import contextlib
import collections

class Config:
    def __init__(self, config_path='config.json'):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                
            # Create required directories
            Path(self.get('paths.temp_audio_dir')).mkdir(exist_ok=True)
            Path(self.get('paths.logs_dir')).mkdir(exist_ok=True)
            
            # Setup logging to both stdout and a file
            log_file = Path(self.get('paths.logs_dir')) / 'bot.log'
            log_level = self.get('logging.level', 'INFO')
            log_format = self.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            logging.basicConfig(
                level=getattr(logging, log_level.upper()),
                format=log_format,
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(log_file)
                ]
            )
            
        except Exception as e:
            print(f"Error loading config: {e}")
            raise

    def get(self, path, default=None):
        """Get configuration value using dot notation"""
        try:
            value = self.config
            for key in path.split('.'):
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate, config):
    """Generates audio frames from PCM audio data."""
    logger = logging.getLogger('frame_generator')
    
    # Configure logger based on config
    if config.get('logging.frame_generator.enabled', False):
        logger.setLevel(getattr(logging, 
            config.get('logging.frame_generator.level', 'INFO').upper()))
    else:
        logger.setLevel(logging.WARNING)
    
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    
    logger.debug(f"Initialized with frame_duration_ms={frame_duration_ms}, sample_rate={sample_rate}")
    
    while offset + n < len(audio):
        frame = Frame(audio[offset:offset + n], timestamp, duration)
        logger.debug(f"Generated frame: timestamp={timestamp:.3f}s, duration={duration:.3f}s")
        yield frame
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames, config):
    """Filters out non-voiced audio frames."""
    logger = logging.getLogger('vad_collector')
    
    # Configure logger based on config
    if config.get('logging.vad_collector.enabled', False):
        logger.setLevel(getattr(logging, 
            config.get('logging.vad_collector.level', 'INFO').upper()))
    else:
        logger.setLevel(logging.WARNING)
    
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    
    logger.debug(f"Initialized with sample_rate={sample_rate}, frame_duration_ms={frame_duration_ms}")
    
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
        logger.debug(f"Frame speech detection: timestamp={frame.timestamp:.3f}s, is_speech={is_speech}")
        
        if not triggered:
            ring_buffer.append(frame)
            num_voiced = len([f for f in ring_buffer if vad.is_speech(f.bytes, sample_rate)])
            logger.debug(f"Not triggered: buffer_size={len(ring_buffer)}, num_voiced={num_voiced}")
            
            if num_voiced > 0.9 * ring_buffer.maxlen:
                logger.info("Voice detected - starting collection")
                triggered = True
                voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append(frame)
            num_unvoiced = len([f for f in ring_buffer if not vad.is_speech(f.bytes, sample_rate)])
            logger.debug(f"Triggered: buffer_size={len(ring_buffer)}, num_unvoiced={num_unvoiced}")
            
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                logger.info("Voice ended - yielding segment")
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []

    if voiced_frames:
        logger.debug("Yielding final voice segment")
        yield b''.join([f.bytes for f in voiced_frames])

class VoiceTranslatorSink(Sink):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger('VoiceTranslatorSink')
        
        # Initialize VAD
        self.vad = webrtcvad.Vad(config.get('voice.vad_aggressiveness'))
        
        # Load Google credentials
        try:
            credentials = service_account.Credentials.from_service_account_file(
                config.get('google.credentials_path'),
                scopes=[
                    'https://www.googleapis.com/auth/cloud-platform',
                    'https://www.googleapis.com/auth/speech',
                    'https://www.googleapis.com/auth/cloud-translation'
                ]
            )
            
            self.speech_client = speech_v1.SpeechClient(credentials=credentials)
            self.translate_client = translate_v2.Client(credentials=credentials)
            
        except Exception as e:
            self.logger.error(f"Error loading Google credentials: {e}")
            raise
        
        # Initialize audio processing parameters
        self.buffer = []
        self.speaking_buffer = []
        self.silence_duration = 0
        self.is_speaking = False
        
        self.RATE = config.get('voice.sample_rate')
        self.CHUNK_DURATION_MS = config.get('voice.chunk_duration_ms')
        self.CHUNK_SIZE = int(self.RATE * self.CHUNK_DURATION_MS / 1000)
        self.SILENCE_THRESHOLD = config.get('voice.silence_threshold')
    
    async def _process_utterance(self, audio_data, user):
        """Process a complete utterance with language autodetection"""
        temp_filename = Path(self.config.get('paths.temp_audio_dir')) / f"temp_{user.id}.wav"
        
        try:
            # Save audio to WAV file
            with wave.open(str(temp_filename), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.RATE)
                wf.writeframes(audio_data)
            
            # Read audio content
            with open(temp_filename, 'rb') as audio_file:
                content = audio_file.read()
            
            # Configure speech recognition with language detection
            audio = speech_v1.RecognitionAudio(content=content)
            config = speech_v1.RecognitionConfig(
                encoding=speech_v1.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.RATE,
                language_code="es-ES",  # Primary language
                alternative_language_codes=["en-US"],  # Alternative language
                model="default"
            )
            
            # Perform speech recognition
            response = self.speech_client.recognize(config=config, audio=audio)
            
            if not response.results:
                return
            
            # Get transcript and detected language
            transcript = response.results[0].alternatives[0].transcript
            detected_language = response.results[0].language_code.split('-')[0]  # Get 'en' or 'es'
            
            # Get target language based on detected language
            translation_config = self.config.get(f'google.translation.{detected_language}')
            if not translation_config:
                self.logger.error(f"Unsupported detected language: {detected_language}")
                return
                
            target_language = translation_config['target']
            
            # Translate text
            translation = self.translate_client.translate(
                transcript,
                target_language=target_language
            )
            
            # Create message with language indicators
            message_content = (
                f"User: {user.name}\n"
                f"Detected Language: {translation_config['name']}\n"
                f"Original: {transcript}\n"
                f"Translated: {translation['translatedText']}"
            )
            
            # Send or update Discord message
            if hasattr(self, 'bot_message'):
                await self.bot_message.edit(content=message_content)
            else:
                self.bot_message = await self.voice_client.channel.send(message_content)
                
        except Exception as e:
            self.logger.error(f"Error processing utterance: {e}")
            if hasattr(self, 'bot_message'):
                await self.bot_message.edit(content=f"Error processing speech: {str(e)}")
            
        finally:
            if temp_filename.exists():
                temp_filename.unlink()

    def write(self, data, user):
        try:
            self.logger.debug(f"Received data from user {user}")
            frames = frame_generator(self.CHUNK_DURATION_MS, data, self.RATE, self.config)
            segments = vad_collector(
                self.RATE, 
                self.CHUNK_DURATION_MS, 
                self.SILENCE_THRESHOLD, 
                self.vad, 
                frames,
                self.config
            )            
            for segment in segments:
                asyncio.create_task(self._process_utterance(segment, user))
                
        except Exception as e:
            self.logger.error(f"Error in write method: {e}, user: {user}")

    def _resample(self, audio_data, orig_rate, target_rate):
        """Resample audio data to target rate"""
        duration = len(audio_data) / orig_rate
        target_length = int(duration * target_rate)
        return np.interp(
            np.linspace(0, duration, target_length),
            np.linspace(0, duration, len(audio_data)),
            audio_data
        ).astype(np.int16)
    
    def cleanup(self):
        """Clean up resources"""
        self.speech_client = None
        self.translate_client = None

class VoiceTranslatorBot(discord.Bot):
    def __init__(self, config_path='config.json'):
        self.config = Config(config_path)
        super().__init__(
            description=self.config.get('bot.description')
        )
        self.logger = logging.getLogger('VoiceTranslatorBot')
        self.connections = {}
    
    async def start_recording(self, voice_client):
        """Start recording and translating voice in a voice channel"""
        sink = VoiceTranslatorSink(self.config)
        voice_client.start_recording(sink, self.finished_callback)
        self.connections[voice_client.guild.id] = voice_client
    
    async def stop_recording(self, guild_id):
        """Stop recording in a specific guild"""
        if guild_id in self.connections:
            voice_client = self.connections[guild_id]
            voice_client.stop_recording()
            del self.connections[guild_id]
    
    async def finished_callback(self, sink, *args):
        """Callback function when recording is finished"""
        self.logger.info("Recording finished")

def setup_commands(bot):
    @bot.slash_command(name="join", description="Join your voice channel and start translating")
    async def join(ctx):
        if not ctx.author.voice:
            await ctx.respond("You need to be in a voice channel!")
            return
        
        voice_client = await ctx.author.voice.channel.connect()
        await bot.start_recording(voice_client)
        await ctx.respond(
            "Joined voice channel and started translating!\n"
            "I will automatically detect whether you speak in English or Spanish "
            "and translate to the other language."
        )

    @bot.slash_command(name="leave", description="Leave the voice channel")
    async def leave(ctx):
        if not ctx.voice_client:
            await ctx.respond("I'm not in a voice channel!")
            return
        
        await bot.stop_recording(ctx.guild.id)
        await ctx.voice_client.disconnect()
        await ctx.respond("Left voice channel!")

if __name__ == "__main__":
    try:
        # Initialize bot with config
        bot = VoiceTranslatorBot('config.json')
        setup_commands(bot)
        
        # Run bot
        bot.run(bot.config.get('bot.token'))
        
    except Exception as e:
        logging.error(f"Error running bot: {e}")