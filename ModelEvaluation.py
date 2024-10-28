from datasets import load_dataset
from WhisperSTTModel import pipe
from EdgeTTSModel import *
from jiwer import wer
import torchaudio
import librosa
import asyncio
import string
import torch
import os

test_data = load_dataset(
    dataset_name, "en", split="validation", streaming=True, trust_remote_code=True)


def resample_audio(batch):
    audio_array = batch["audio"]["array"]
    sampling_rate = batch["audio"]["sampling_rate"]

    if sampling_rate == 16000:
        return batch

    audio_tensor = torch.tensor(audio_array, dtype=torch_dtype).unsqueeze(0)
    resampler = torchaudio.transforms.Resample(
        orig_freq=sampling_rate, new_freq=16000)
    resampled_audio_tensor = resampler(audio_tensor)
    resampled_audio = resampled_audio_tensor.squeeze(0).numpy()

    batch["audio"]["array"] = resampled_audio
    batch["audio"]["sampling_rate"] = 16000

    return batch


def transcribe_audio(file_path):
    audio_array, _ = librosa.load(file_path, sr=16000)
    result = pipe(audio_array)
    return result['text'].lower()


async def testing_TTS(test_data, n_samples):
    total_wer = 0
    count = 0
    for example in test_data:
        if count >= n_samples:
            break

        original_text = example['sentence'].translate(
            str.maketrans('', '', string.punctuation)).lower()
        predicted_text = pipe(example["audio"]["array"])["text"]
        predicted_text = predicted_text.translate(
            str.maketrans('', '', string.punctuation)).lower()
        error_rate = wer(original_text, predicted_text)

        if error_rate == 0:
            audio_path = await synthesize_speech(original_text)
            transcribed_text = transcribe_audio(audio_path)
            transcribed_text = transcribed_text.translate(
                str.maketrans('', '', string.punctuation)).lower()
            word_wer = wer(original_text, transcribed_text)
            total_wer += word_wer
            count += 1

            renamed_output_file = f"test_TTS_{count}.mp3"
            os.rename(audio_path, renamed_output_file)
            print(f"Original Sentence: {original_text}")
            print(f"Transcribed Sentence: {transcribed_text}")
            print(f"WER: {word_wer * 100:.2f}%")

    average_wer = total_wer / n_samples
    print(f"Average WER for {n_samples} samples: {average_wer * 100:.2f}%")


test_data = test_data.map(resample_audio)
test_data = test_data.map(
    lambda x: {k: v for k, v in x.items() if k not in columns_to_remove})

asyncio.run(testing_TTS(test_data, 20))
