import os
import csv
from pydub import AudioSegment
from torch.utils.data import Dataset

# returns AudioSegment tuple
class SongOkeCroppedDataset(Dataset):
    def __init__(self, root_dir, list_file='train.csv', bi_transform=None):
        self.root_dir = root_dir

        csv_path = os.path.join(root_dir, list_file)
        with open(csv_path, 'r') as fp:
            reader = csv.reader(fp)

            samples = []
            for row in reader:
                samples.append([ int(row[0]), row[1], row[2] ])

        self.samples = samples
        self.bi_transform = bi_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample_index, song_relpath, oke_relpath = self.samples[index]

        song_path = os.path.join(self.root_dir, song_relpath)
        oke_path = os.path.join(self.root_dir, oke_relpath)

        def load_audio(path):
            ext = os.path.splitext(path)[1][1:] # FILE.m4a -> .m4a -> m4a
            return AudioSegment.from_file(path, format=ext)

        song = load_audio(song_path)
        oke = load_audio(oke_path)

        if self.bi_transform is not None:
            song, oke = self.bi_transform(song, oke)

        return song, oke

if __name__ == '__main__':
    root_dir = os.path.expanduser('~/datasets/MaouSongOkeCroppedDataset')

    dataset = SongOkeCroppedDataset(root_dir)

    song, oke = dataset[0]

    # num of samples
    # _data is raw bytes of wave
    print(len(song.raw_data) / song.sample_width)
    print(len(oke.raw_data) / oke.sample_width)

    # 224x224x3 = 150528
