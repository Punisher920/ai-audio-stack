"""OpenAI Audio API - Transcription & Text-to-Speech

For HealthAIde, StorySpark, and your other projects.
"""

import os
from pathlib import Path
from openai import OpenAI, AsyncOpenAI


class OpenAIAudio:
    """Unified OpenAI Audio handler for transcription and TTS."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
    
    # === TRANSCRIPTION (Speech-to-Text) ===
    
    def transcribe(self, audio_path: str, model: str = "gpt-4o-transcribe") -> str:
        """Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file (mp3, wav, m4a, etc.)
            model: Model to use (gpt-4o-transcribe or gpt-4o-mini-transcribe)
        
        Returns:
            Transcribed text
        """
        with open(audio_path, "rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                file=audio_file,
                model=model
            )
        return response.text
    
    async def transcribe_async(self, audio_path: str, model: str = "gpt-4o-transcribe") -> str:
        """Async version of transcribe."""
        with open(audio_path, "rb") as audio_file:
            response = await self.async_client.audio.transcriptions.create(
                file=audio_file,
                model=model
            )
        return response.text
    
    # === TEXT-TO-SPEECH ===
    
    def text_to_speech(
        self,
        text: str,
        output_path: str = "output.wav",
        voice: str = "coral",
        model: str = "gpt-4o-mini-tts",
        response_format: str = "wav"
    ) -> str:
        """Convert text to speech audio file.
        
        Args:
            text: Text to convert to speech
            output_path: Where to save the audio file
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer, coral, etc.)
            model: TTS model (gpt-4o-mini-tts or gpt-4o-tts)
            response_format: Output format (mp3, opus, aac, flac, wav, pcm)
        
        Returns:
            Path to saved audio file
        """
        response = self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format=response_format
        )
        
        # Write audio to file
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        return output_path
    
    async def text_to_speech_async(
        self,
        text: str,
        output_path: str = "output.wav",
        voice: str = "coral",
        model: str = "gpt-4o-mini-tts",
        response_format: str = "wav"
    ) -> str:
        """Async version of text_to_speech."""
        response = await self.async_client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format=response_format
        )
        
        content = await response.aread()
        with open(output_path, "wb") as f:
            f.write(content)
        
        return output_path


# === USAGE EXAMPLES ===

if __name__ == "__main__":
    # Initialize
    audio = OpenAIAudio()
    
    # Example 1: Transcribe audio
    text = audio.transcribe("interview.mp3")
    print(f"Transcription: {text}")
    
    # Example 2: Generate speech
    audio.text_to_speech(
        text="Welcome to HealthAIde. How can I help you today?",
        output_path="greeting.wav",
        voice="coral"
    )
    
    # Example 3: Voice loop (transcribe -> process -> speak)
    user_audio = "user_question.wav"
    transcription = audio.transcribe(user_audio)
    
    # Process with your GPT logic here
    answer = f"You said: {transcription}"
    
    audio.text_to_speech(answer, "response.wav")
