import os
import sys
import shutil
import csv
import tempfile
from tqdm import tqdm
from SongOkeFullDataset import *

if __name__ == '__main__':
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]

    os.makedirs(out_dir, exist_ok=True)

    dataset = SongOkeFullDataset(root_dir=os.path.expanduser(data_dir))
    csv_path = os.path.join(out_dir, 'audios.csv')

    with open(csv_path, 'w') as csv_fp:
        writer = csv.writer(csv_fp)

        for audio_index, (song, oke) in tqdm(enumerate(dataset), total=len(dataset)):
            song_millis = len(song)
            oke_millis = len(oke)
            min_millis = min(song_millis, oke_millis)

            writer.writerow([ str(audio_index), dataset.song_files[audio_index], dataset.oke_files[audio_index] ])

            time_grids = range(0, min_millis, 5000)

            for sample_index, start_millis in tqdm(enumerate(time_grids), total=len(time_grids)):
                song_cropped = song[start_millis:start_millis+5000]
                oke_cropped = oke[start_millis:start_millis+5000]
                if len(song_cropped) != len(oke_cropped) or len(song_cropped) != 5000:
                    continue

                song_dir_path = os.path.join(out_dir, 'song', '%d' % audio_index)
                song_path = os.path.join(song_dir_path, '%d.wav' % sample_index)
                oke_dir_path = os.path.join(out_dir, 'oke', '%d' % audio_index)
                oke_path = os.path.join(oke_dir_path, '%d.wav' % sample_index)

                os.makedirs(song_dir_path, exist_ok=True)
                os.makedirs(oke_dir_path, exist_ok=True)

                with tempfile.NamedTemporaryFile(suffix='.wav') as fp:
                    song_cropped.export(fp, format='wav')
                    shutil.copy(fp.name, song_path)
                with tempfile.NamedTemporaryFile(suffix='.wav') as fp:
                    oke_cropped.export(fp, format='wav')
                    shutil.copy(fp.name, oke_path)
