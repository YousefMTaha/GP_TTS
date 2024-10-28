from huggingface_hub import login
import edge_tts
import warnings


warnings.filterwarnings("ignore")
login("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")


async def synthesize_speech(text, voice="en-US-JennyNeural") -> str:
    communicate = edge_tts.Communicate(text, voice)
    print("voice is generated")
    path = f"GP-TTS\Generated_Voices\output_{voice}.mp3"
    await communicate.save(path)
    return path, f"output_{voice}.mp3"
