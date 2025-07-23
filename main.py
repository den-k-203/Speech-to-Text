from service.diarization_service import process_audio_segments

path='audio/conversation.mp3'

if __name__ == '__main__':
    process_audio_segments(path_to_audio=path, output_json_path="conversation.json")
