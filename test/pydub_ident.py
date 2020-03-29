import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from pydub import AudioSegment
import BiAudioTransform.functional as BTF

path = os.path.expanduser('~/datasets/MaouSongOkeCroppedDataset/song/0/14.wav')
audio = AudioSegment.from_file(path, format='wav')

audio = audio.set_frame_rate(44100)
audio = audio.set_sample_width(4)

tensor = BTF.to_tensor(audio)
audio = BTF.to_pydub(tensor)

audio.export('export.wav')
