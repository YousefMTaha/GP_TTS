from datasets import load_dataset
from EdgeTTSModel import *
import torchaudio
import evaluate
import string
import torch

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


test_data = test_data.map(resample_audio)
test_data = test_data.map(
    lambda x: {k: v for k, v in x.items() if k not in columns_to_remove})

wer_metric = evaluate.load("wer")


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def compute_wer(dataset, num_of_samples: int):
    references = []
    predictions = []
    total_samples = 0

    for sample in dataset:
        if total_samples > num_of_samples:
            break

        prediction = pipe(sample["audio"]["array"])["text"]
        refrence = sample["sentence"]

        prediction = prediction.translate(
            str.maketrans('', '', string.punctuation)).lower()
        refrence = refrence.translate(
            str.maketrans('', '', string.punctuation)).lower()

        predictions.append(prediction)
        references.append(refrence)
        word_wer = wer_metric.compute(
            predictions=[prediction], references=[refrence])

        print(f"Actaul Sentence: {refrence}")
        print(f"Predicted Sentence: {prediction}")
        print(f"Word Error Rate (WER): {word_wer * 100:.2f}%")
        total_samples += 1

    wer = wer_metric.compute(predictions=predictions, references=references)
    return wer


wer_score = compute_wer(test_data, 50)
print(f"Total Word Error Rate (WER): {wer_score * 100:.2f}%")
