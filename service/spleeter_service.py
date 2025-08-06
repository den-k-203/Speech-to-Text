import os
from spleeter.separator import Separator

def extract_vocals(input_path: str, output_dir: str) -> str:
    separator = Separator('spleeter:2stems')
    separator.separate_to_file(input_path, output_dir)
    file_name = os.path.splitext(os.path.basename(input_path))[0]

    vocals_path = os.path.join(output_dir, file_name, 'vocals.wav')

    if not os.path.exists(vocals_path):
        raise FileNotFoundError(f"Не знайдено файл: {vocals_path}")

    return vocals_path
