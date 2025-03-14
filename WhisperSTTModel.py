from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from huggingface_hub import login
import warnings
import torch
import os

warnings.filterwarnings("ignore")
login("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")

os.environ["WANDB_DISABLED"] = "true"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float32
model_name = "openai/whisper-large-v3-turbo"
dataset_name = "mozilla-foundation/common_voice_17_0"
const_inputs = "input_values"
const_input_ids = "input_ids"
const_labels = "labels"
columns_to_remove = ["client_id", "up_votes", "down_votes",
                     "age", "gender", "accent", "locale", "segment", "variant"]


def init_model_and_processor():
    processor = AutoProcessor.from_pretrained(model_name)

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True,
    )

    model.to(device)

    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    return model, processor


def get_pipeline(model, processor):
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )


model, processor = init_model_and_processor()
pipe = get_pipeline(model, processor)
