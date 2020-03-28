import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import time
from pydub import AudioSegment
import BiAudioTransform as BT

song_path = os.path.expanduser('~/datasets/MaouSongOkeDataset/song/song_kouichi_the_milky_way.m4a')
oke_path = os.path.expanduser('~/datasets/MaouSongOkeDataset/oke/oke_song_kouichi_the_milky_way.m4a')

song = AudioSegment.from_file(song_path, format='m4a')
oke = AudioSegment.from_file(oke_path, format='m4a')

to_tensor = BT.ToTensor()

ts = time.time()
song, oke = to_tensor(song, oke)
elapsed = time.time() - ts

print('Elapsed: %f s' % elapsed)

print(type(song), song.dtype, song.shape)
print(type(oke), oke.dtype, oke.shape)
