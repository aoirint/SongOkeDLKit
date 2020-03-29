import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from pydub import AudioSegment

'''
in: pydub.AudioSegment
out: torch.Tensor (float32)
'''
def to_tensor(audio):
    sample_width = audio.sample_width
    sample_bits = 8 * sample_width
    sample_max_int = 2 ** (sample_bits - 1)
    sample_channels = audio.channels

    samples = np.asarray(audio.get_array_of_samples())
    samples = samples.reshape((-1, 2)).transpose((1, 0)) # LRLR -> Channel, Samples

    samples = samples.astype(np.float64) / sample_max_int
    samples = samples.astype(np.float32)

    samples = torch.from_numpy(samples).type(torch.float32)

    return samples

'''
in: torch.Tensor (float32)
out: pydub.AudioSegment
'''
def to_pydub(tensor, sample_width=4, frame_rate=44100):
    # assert len(tensor.shape) == 2 # [channels, samples]

    samples = tensor.cpu().type(torch.float32).numpy().astype(np.float32)
    samples = np.clip(samples, -1.0, 1.0)

    sample_bits = 8 * sample_width
    sample_max_int = 2 ** (sample_bits - 1)
    channels = samples.shape[0]

    samples = (samples * sample_max_int).astype(np.int32)
    samples = samples.transpose((1, 0)).reshape((-1, )) # Channel, Samples -> LRLR
    # samples = samples.astype('>i4') # big endian 4 bytes signed int
    # samples = samples.astype('<i4') # little endian 4 bytes signed int

    output = AudioSegment(
        samples.tobytes(),
        sample_width=sample_width,
        frame_rate=frame_rate,
        channels=channels,
    )
    return output
