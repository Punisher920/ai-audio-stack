"""ElevenLabs Text-to-Speech API

Premium voice synthesis for your apps - HealthAIde, StorySpark, dentist assistant.
"""

import os
from elevenlabs.client import ElevenLabs


class ElevenLabsTTS:
    """ElevenLabs premium voice handler for high-quality TTS."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.client = ElevenLabs(api_key=self.api_key)
    
    def text_to_speech(
        self,
        text: str,
        output_path: str = "output.mp3",
        voice_id: str = None,
        model_id: str = "eleven_turbo_v2_5",
        output_format: str = "mp3_44100_128"
    ) -> str:
        """Convert text to premium speech audio.
        
        Args:
            text: Text to convert to speech
            output_path: Where to save the audio file
            voice_id: ElevenLabs voice ID from your dashboard
            model_id: Model to use (eleven_turbo_v2_5, eleven_multilingual_v2, etc.)
            output_format: Format (mp3_44100_128, pcm_44100, etc.)
        
        Returns:
            Path to saved audio file
        """
        # Use default voice if none specified
        if not voice_id:
            voice_id = self.get_default_voice()
        
        audio = self.client.text_to_speech.convert(
            voice_id=voice_id,
            model_id=model_id,
            text=text,
            output_format=output_format,
        )
        
        with open(output_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)
        
        return output_path
    
    def get_default_voice(self) -> str:
        """Get the first available voice ID from your account."""
        voices = self.client.voices.get_all()
        if voices.voices:
            return voices.voices[0].voice_id
        raise ValueError("No voices available in your ElevenLabs account")
    
    def list_voices(self) -> list:
        """List all available voices in your account."""
        voices = self.client.voices.get_all()
        return [
            {
                "voice_id": v.voice_id,
                "name": v.name,
                "category": v.category if hasattr(v, 'category') else None
            }
            for v in voices.voices
        ]
    
    def text_to_speech_streaming(
        self,
        text: str,
        voice_id: str = None,
        model_id: str = "eleven_turbo_v2_5"
    ):
        """Stream audio generation (for real-time playback).
        
        Returns:
            Generator yielding audio chunks
        """
        if not voice_id:
            voice_id = self.get_default_voice()
        
        return self.client.text_to_speech.convert(
            voice_id=voice_id,
            model_id=model_id,
            text=text,
            output_format="mp3_44100_128",
        )


# === USAGE EXAMPLES ===

if __name__ == "__main__":
    # Initialize
    tts = ElevenLabsTTS()
    
    # Example 1: Simple text-to-speech
    tts.text_to_speech(
        text="Welcome to your health assistant. Let me know how I can help.",
        output_path="welcome.mp3"
    )
    
    # Example 2: List available voices
    voices = tts.list_voices()
    print(f"Available voices: {len(voices)}")
    for v in voices[:3]:  # Show first 3
        print(f"  - {v['name']} (ID: {v['voice_id']})")
    
    # Example 3: Use specific voice
    if voices:
        tts.text_to_speech(
            text="This is a custom voice demonstration.",
            output_path="custom_voice.mp3",
            voice_id=voices[0]['voice_id']
        )
    
    # Example 4: Streaming for real-time playback
    # stream = tts.text_to_speech_streaming("Hello world")
    # for chunk in stream:
    #     # Send chunk to audio player or save incrementally
    #     pass
