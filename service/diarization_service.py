import json
import os
import tempfile
import torchaudio
import soundfile as sf
import logging
import torch

from pathlib import Path
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from domain.Speaker import Speaker
from utils.logger_config import setup_logger
from config import LOCAL_PATH_TO_MODEL

logger = setup_logger()
logger.success = lambda msg: logger.log(logging.INFO, f"\033[1;32m{msg}\033[0m")

WHISPER_SAMPLE_RATE = 16000


def init_diarization_pipeline():
    logger.info("Ініціалізація діаризаційного пайплайна...")
    pipeline = Pipeline.from_pretrained(Path(LOCAL_PATH_TO_MODEL))
    pipeline.to(torch.device("cuda"))
    return pipeline


def init_whisper_model():
    logger.info("Ініціалізація моделі Whisper...")
    model = WhisperModel("base", device="cuda", compute_type="float16")
    logger.info("Whisper модель ініціалізована")
    return model


def load_audio(path_to_audio):
    audio, sr = torchaudio.load(path_to_audio)
    logger.info(f"Завантажено аудіо: {path_to_audio} (sample rate: {sr})")
    if sr != WHISPER_SAMPLE_RATE:
        logger.info(f"Ресемплінг з {sr} до {WHISPER_SAMPLE_RATE}")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=WHISPER_SAMPLE_RATE)
        audio = resampler(audio)
        sr = WHISPER_SAMPLE_RATE
    return audio[0].numpy(), sr


def transcribe(audio_segment, sr, whisper_model):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sf.write(tmp_path, audio_segment, samplerate=sr)
        segments, _ = whisper_model.transcribe(tmp_path)
        return "".join([seg.text for seg in segments]).strip()
    except Exception as e:
        logger.error(f"❌ Помилка при обробці сегмента: {e}")
        return ""
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def process_single_turn(turn, speaker_label, full_audio_np, sr, whisper_model):
    start = int(turn.start * sr)
    end = int(turn.end * sr)
    duration = (end - start) / sr

    logger.debug(f"Спікер {speaker_label}: {turn.start:.2f}s — {turn.end:.2f}s (тривалість: {duration:.2f}s)")

    if duration < 0.3 or start >= end or end > len(full_audio_np):
        logger.warning("Пропущено сегмент: некоректна довжина або позиція.")
        return None

    audio_segment = full_audio_np[start:end]

    if audio_segment.size == 0 or audio_segment.ndim != 1:
        logger.warning(f"Неправильний сегмент (size: {audio_segment.size}, shape: {audio_segment.shape})")
        return None

    audio_segment = audio_segment.astype("float32")
    logger.debug(f"Segment shape: {audio_segment.shape}, dtype: {audio_segment.dtype}")

    text = transcribe(audio_segment, sr, whisper_model)
    logger.info(f"[{speaker_label}] {text}")

    return Speaker(speaker_label, turn.start, turn.end, text).to_dict()


def diarize_audio(path_to_audio):
    logger.info(f"Старт діаризації для: {path_to_audio}")
    pipeline = init_diarization_pipeline()
    diarization = pipeline(path_to_audio)
    logger.info("Завершено діаризацію.")
    return diarization


def process_audio_segments(path_to_audio, output_json_path="output.json"):
    if not os.path.exists(path_to_audio):
        logger.error(f"Файл {path_to_audio} не знайдено.")
        raise FileNotFoundError(f"Файл {path_to_audio} не знайдено.")

    diarization = diarize_audio(path_to_audio)
    whisper_model = init_whisper_model()
    full_audio_np, sr = load_audio(path_to_audio)

    results = []
    idx = 0

    for turn, _, speaker_label in diarization.itertracks(yield_label=True):
        speaker_data = process_single_turn(turn, speaker_label, full_audio_np, sr, whisper_model)
        if speaker_data:
            results.append(speaker_data)
            idx += 1

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    logger.success(f"✅ Збережено результат у файл: {output_json_path}")
    logger.info(f"Оброблено сегментів: {idx}")
