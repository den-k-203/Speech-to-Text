class Speaker:
    def __init__(self, speaker, start, end, text):
        self.speaker = speaker
        self.start = round(start, 2)
        self.end = round(end, 2)
        self.text = text.strip()

    def to_dict(self):
        return {
            "speaker": self.speaker,
            "start": self.start,
            "end": self.end,
            "text": self.text
        }
