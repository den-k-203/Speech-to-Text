from service.diarization_service import process_audio_segments
from service.spleeter_service import extract_vocals


path='audio/news_tsn.mp3'
output_dir_path = 'audio/vocal'

if __name__ == '__main__':
    # process_audio_segments(path_to_audio=path, output_json_path="audio/json/news_tsn.json")
    extract_vocals(input_path=path, output_dir=output_dir_path)