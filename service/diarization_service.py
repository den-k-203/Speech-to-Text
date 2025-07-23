import json
import os
import tempfile
import torchaudio
import soundfile as sf
import logging

import torch
from pyannote.audio import Pipeline
from config import token_huggingface
from faster_whisper import WhisperModel
from domain.Speaker import Speaker
from utils.logger_config import setup_logger

logger = setup_logger()

logger.success = lambda msg: logger.log(logging.INFO, f"\033[1;32m{msg}\033[0m")

WHISPER_SAMPLE_RATE = 16000
DIARIZATION_PIPELINE_ID = "pyannote/speaker-diarization"

def init_diarization_pipeline():
    logger.info("Ініціалізація діаризаційного пайплайна...")

    pipeline = Pipeline.from_pretrained(
        DIARIZATION_PIPELINE_ID,
        use_auth_token=token_huggingface
    )

    pipeline.to(torch.device("cuda"))

    return pipeline

def init_whisper_model():
    logger.info("Ініціалізація моделі Whisper...")
    model = WhisperModel("base", device="cuda", compute_type="float16")
    logger.info(f"Whisper модель ініціалізована")
    return model

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

    results = []
    idx = 0

    full_audio, sr = torchaudio.load(path_to_audio)
    logger.info(f"Завантажено аудіо: {path_to_audio} (sample rate: {sr})")

    if sr != WHISPER_SAMPLE_RATE:
        logger.info(f"Ресемплінг з {sr} до {WHISPER_SAMPLE_RATE}")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=WHISPER_SAMPLE_RATE)
        full_audio = resampler(full_audio)
        sr = WHISPER_SAMPLE_RATE

    full_audio_np = full_audio[0].numpy()

    for turn, _, speaker_label in diarization.itertracks(yield_label=True):
        start = int(turn.start * sr)
        end = int(turn.end * sr)
        duration = (end - start) / sr

        logger.debug(f"Спікер {speaker_label}: {turn.start:.2f}s — {turn.end:.2f}s (тривалість: {duration:.2f}s)")

        if duration < 0.3:
            logger.warning("Пропущено сегмент: занадто короткий.")
            continue

        audio_segment = full_audio_np[start:end]

        if audio_segment.size == 0:
            logger.warning("Пропущено сегмент: порожній аудіофрагмент.")
            continue

        if audio_segment.ndim != 1:
            logger.warning(f"Сегмент має неправильну форму: {audio_segment.shape}")
            continue

        audio_segment = audio_segment.astype("float32")
        logger.debug(f"Segment shape: {audio_segment.shape}, dtype: {audio_segment.dtype}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            sf.write(tmp_path, audio_segment, samplerate=sr)
            segments, _ = whisper_model.transcribe(tmp_path)
        except Exception as e:
            logger.error(f"❌ Помилка при обробці сегмента: {e}")
            continue
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        text = "".join([seg.text for seg in segments]).strip()
        logger.info(f"[{speaker_label}] {text}")

        speaker_instance = Speaker(speaker_label, turn.start, turn.end, text)
        results.append(speaker_instance.to_dict())
        idx += 1

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    logger.success(f"✅ Збережено результат у файл: {output_json_path}")
    logger.info(f"Оброблено сегментів: {idx}")
