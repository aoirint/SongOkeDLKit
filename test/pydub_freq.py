import os
from pydub import AudioSegment

path = os.path.expanduser('~/datasets/MaouSongOkeCroppedDataset/song/0/14.wav')
audio = AudioSegment.from_file(path, format='wav')

print(audio.frame_rate)
print(audio.sample_width)

audio = audio.set_frame_rate(44100)
audio = audio.set_sample_width(4)

audio.export('export.wav')
