import random
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import .functional as BTF

class BiAudioTransform:
    pass

'''
in: pydub.AudioSegment
out: torch.Tensor (float32)
'''
class ToTensor(BiAudioTransform):
    def __call__(self, song, oke):
        return BTF.to_tensor(song), BTF.to_tensor(oke)

'''
in: torch.Tensor
out: torch.Tensor
'''
class RandomCrop(BiAudioTransform):
    def __init__(self, size):
        self.size = size

    def __call__(self, song, oke):
        assert song.shape[1] == oke.shape[1] # num of samples
        num_samples = song.shape[1]

        start = random.randrange(0, num_samples-self.size)
        end = start + self.size

        song = BTF.crop(song, start=start, end=end)
        oke = BTF.crop(oke, start=start, end=end)

        return song, oke
