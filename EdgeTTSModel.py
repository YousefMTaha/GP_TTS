from huggingface_hub import login
import edge_tts
import warnings
import torch
import os

warnings.filterwarnings("ignore")
login("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")

os.environ["WANDB_DISABLED"] = "true"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float32
dataset_name = "mozilla-foundation/common_voice_17_0"
columns_to_remove = ["client_id", "up_votes", "down_votes",
                     "age", "gender", "accent", "locale", "segment", "variant"]


async def synthesize_speech(text, voice="en-US-JennyNeural") -> str:
    communicate = edge_tts.Communicate(text, voice)
    path = f"GP-TTS\Generated_Voices\output_{voice}.mp3"
    file_path = os.path.abspath(path)
    await communicate.save(file_path)
    return file_path