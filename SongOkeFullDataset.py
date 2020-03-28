import os
from pydub import AudioSegment
from torch.utils.data import Dataset

# returns AudioSegment tuple
class SongOkeFullDataset(Dataset):
    def __init__(self, root_dir, bi_transform=None):
        self.root_dir = root_dir
        self.song_dir_path = os.path.join(root_dir, 'song')
        self.oke_dir_path = os.path.join(root_dir, 'oke')

        self.song_files = sorted(os.listdir(self.song_dir_path))
        self.oke_files = [ 'oke_' + song_file for song_file in self.song_files ]

        for oke_file in self.oke_files:
            oke_path = os.path.join(self.oke_dir_path, oke_file)
            if not os.path.exists(oke_path):
                raise Exception('Not found oke file:', oke_file)

        self.bi_transform = bi_transform

    def __len__(self):
        return len(self.song_files)

    def __getitem__(self, index):
        song_file = self.song_files[index]
        oke_file = self.oke_files[index]

        song_path = os.path.join(self.song_dir_path, song_file)
        oke_path = os.path.join(self.oke_dir_path, oke_file)

        def load_audio(path):
            ext = os.path.splitext(path)[1][1:] # FILE.m4a -> .m4a -> m4a
            return AudioSegment.from_file(path, format=ext)

        song = load_audio(song_path)
        oke = load_audio(oke_path)

        if self.bi_transform is not None:
            song, oke = self.bi_transform(song, oke)

        return song, oke

if __name__ == '__main__':
    root_dir = os.path.expanduser('~/datasets/MaouSongOkeDataset')

    dataset = SongOkeFullDataset(root_dir)

    song, oke = dataset[0]

    # num of samples
    # _data is raw bytes of wave
    print(len(song.raw_data) / song.sample_width)
    print(len(oke.raw_data) / oke.sample_width)

    # 224x224x3 = 150528
