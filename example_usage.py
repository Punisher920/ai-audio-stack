"""Complete Audio Stack Usage Examples

Shows how to use OpenAI Audio + ElevenLabs together for your projects.
"""

import os
from openai_audio import OpenAIAudio
from elevenlabs_tts import ElevenLabsTTS
from openai import OpenAI


def example_1_voice_assistant():
    """Voice assistant: Listen -> Think -> Speak (OpenAI STT + GPT + ElevenLabs TTS)"""
    print("\n=== Example 1: Voice Assistant ===")
    
    # Initialize
    audio = OpenAIAudio()
    tts = ElevenLabsTTS()
    gpt = OpenAI()
    
    # Step 1: Transcribe user audio
    user_question = audio.transcribe("user_input.wav")
    print(f"User said: {user_question}")
    
    # Step 2: Get AI response
    response = gpt.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_question}]
    )
    answer = response.choices[0].message.content
    print(f"Assistant: {answer}")
    
    # Step 3: Speak with ElevenLabs premium voice
    tts.text_to_speech(answer, "assistant_reply.mp3")
    print("✓ Response saved to assistant_reply.mp3")


def example_2_healthaide():
    """HealthAIde: Medical assistant with premium voice"""
    print("\n=== Example 2: HealthAIde Medical Assistant ===")
    
    audio = OpenAIAudio()
    tts = ElevenLabsTTS()
    gpt = OpenAI()
    
    # Transcribe patient question
    patient_audio = audio.transcribe("patient_question.wav")
    print(f"Patient: {patient_audio}")
    
    # Get medical information (with safety disclaimers)
    system_prompt = (
        "You are a health information assistant. "
        "Provide helpful information but always remind users to consult healthcare professionals."
    )
    
    response = gpt.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": patient_audio}
        ]
    )
    
    answer = response.choices[0].message.content
    
    # Use ElevenLabs for professional medical voice
    tts.text_to_speech(answer, "health_response.mp3")
    print(f"✓ Health info ready: health_response.mp3")


def example_3_compare_voices():
    """Compare OpenAI vs ElevenLabs TTS"""
    print("\n=== Example 3: Voice Comparison ===")
    
    audio = OpenAIAudio()
    tts = ElevenLabsTTS()
    
    text = "Welcome to your personalized audio experience. This is a test of voice quality."
    
    # OpenAI TTS (fast, good quality)
    audio.text_to_speech(text, "openai_voice.wav", voice="coral")
    print("✓ OpenAI voice saved")
    
    # ElevenLabs TTS (premium, character voices)
    tts.text_to_speech(text, "elevenlabs_voice.mp3")
    print("✓ ElevenLabs voice saved")
    
    print("\nCompare both files to choose what works for your app!")


def example_4_storyspark_narration():
    """StorySpark: Generate story narration"""
    print("\n=== Example 4: StorySpark Children's Book Narration ===")
    
    tts = ElevenLabsTTS()
    gpt = OpenAI()
    
    # Generate story
    story_prompt = "Write a short bedtime story about a friendly dragon, 3 sentences."
    response = gpt.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": story_prompt}]
    )
    
    story = response.choices[0].message.content
    print(f"Story: {story}")
    
    # Narrate with child-friendly voice
    tts.text_to_speech(story, "bedtime_story.mp3")
    print("✓ Story narration ready for QR code")


def example_5_batch_processing():
    """Process multiple files efficiently"""
    print("\n=== Example 5: Batch Audio Processing ===")
    
    audio = OpenAIAudio()
    
    audio_files = ["interview1.mp3", "interview2.mp3", "interview3.mp3"]
    
    for i, file in enumerate(audio_files, 1):
        if os.path.exists(file):
            transcript = audio.transcribe(file)
            
            # Save transcript
            with open(f"transcript_{i}.txt", "w") as f:
                f.write(transcript)
            
            print(f"✓ Processed {file}")
        else:
            print(f"⚠ Skipped {file} (not found)")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  AI AUDIO STACK - Complete Examples")
    print("  OpenAI + ElevenLabs Integration")
    print("="*50)
    
    # Run examples (comment out if files don't exist yet)
    # example_1_voice_assistant()
    # example_2_healthaide()
    example_3_compare_voices()
    # example_4_storyspark_narration()
    # example_5_batch_processing()
    
    print("\n✓ All examples complete!")
    print("\nNext steps:")
    print("1. Set OPENAI_API_KEY and ELEVENLABS_API_KEY environment variables")
    print("2. Install: pip install -r requirements.txt")
    print("3. Run: python example_usage.py")
