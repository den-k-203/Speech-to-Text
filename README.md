# 🗣️ Speech-to-Text з діаризацією спікерів

Цей проєкт виконує **розпізнавання мовлення з аудіофайлу** з **розділенням за спікерами** (діаризацією), використовуючи сучасні моделі:
- [`pyannote-audio`](https://github.com/pyannote/pyannote-audio) для діаризації
- [`faster-whisper`](https://github.com/guillaumekln/faster-whisper) для трансформації мовлення в текст
- Підтримка **GPU (CUDA)** для пришвидшення обробки

---

## ⚙️ Вимоги

- **Python**: `3.8` або `3.9`
- **CUDA** (опціонально, для GPU): рекомендовано CUDA `11.8` або `12.1`
- GPU з підтримкою FP16 або INT8 (наприклад, NVIDIA GTX 1650 Ti або новіша)

---

## 📦 Встановлення

1. 🔻 Клонування проєкту:

```bash
git clone https://github.com/your-username/speech-diarization-whisper.git
cd speech-diarization-whisper
```
2. 🐍 Створіть віртуальне середовище:
```bash
python -m venv .venv
source .venv/bin/activate  # або .venv\Scripts\activate на Windows
```
3. 📥 Встановіть залежності:
```bash
pip install -r requirements.txt
```
Якщо плануєте використовувати GPU, переконайтесь, що torch встановлено з підтримкою CUDA:
```bash
pip uninstall torch torchaudio torchvision -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
---
## 📁 Завантаження моделі локально (офлайн)
Щоб працювати без інтернету та токена Hugging Face, попередньо завантажте модель:
```bash
git lfs install
git clone https://huggingface.co/pyannote/speaker-diarization path/to/saving
```
У `.env` вкажіть змінну LOCAL_PATH_TO_MODEL, як описано нище. Модель буде використана напряму з диску.

---
## 📄 Конфігурація
У файл `.env` додайте ваш шлях до `config.yml` speaker-diarization:
```bash
# Працюєте повністю локально:
LOCAL_PATH_TO_MODEL=<'LOCAL_PATH_TO_MODEL'>
```
---
## 🚀 Запуск
Скрипт для обробки аудіофайлу:
```bash
python main.py
```
У `main.py` викликається функція:
```bash
process_audio_segments("шлях/до/аудіо.wav")
```
Результат зберігається у `output.json` у форматі:
```bash
[
  {
    "speaker": "SPEAKER_00",
    "start": 0.13,
    "end": 4.51,
    "text": "Привіт, як справи?"
  },
  ...
]
```
---
## 🧠 Для чого цей проєкт?
Цей інструмент призначений для:

- 🔊 Автоматичного розпізнавання голосових записів
- 👥 Визначення хто і коли говорить
- 🧾 Формування структурованого транскрипту з таймкодами

Підходить для:

- Записів нарад
- Інтерв’ю
- Подкастів
- Аналізу відео/аудіо-архівів