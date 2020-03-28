import os
import sys
import shutil
import random
import csv
import tempfile
from tqdm import tqdm

if __name__ == '__main__':
    root_dir = sys.argv[1]
    song_root_dir = os.path.join(root_dir, 'song')
    oke_root_dir = os.path.join(root_dir, 'oke')

    seed = 1234
    rand = random.Random(seed)

    num_val = 5
    test_rate = .1

    song_dirs = sorted(os.listdir(song_root_dir), key=lambda file: int(file))
    val_song_dirs = song_dirs[:num_val]
    traintest_song_dirs = song_dirs[num_val:]

    def findall(song_dirs):
        song_path_list = []
        oke_path_list = []
        for song_dir in song_dirs:
            song_dir_path = os.path.join(song_root_dir, song_dir)
            oke_dir_path = os.path.join(oke_root_dir, song_dir)

            song_files = sorted(os.listdir(song_dir_path), key=lambda file: int(os.path.splitext(file)[0]))
            for song_file in song_files:
                song_path = os.path.join(song_dir_path, song_file)
                oke_path = os.path.join(oke_dir_path, song_file)
                assert os.path.exists(oke_path)

                song_path_list.append(song_path)
                oke_path_list.append(oke_path)
        return song_path_list, oke_path_list

    val_song_path_list, val_oke_path_list = findall(val_song_dirs)
    traintest_song_path_list, traintest_oke_path_list = findall(traintest_song_dirs)

    n_traintest_samples = len(traintest_song_path_list)
    traintest_indices = rand.sample(range(n_traintest_samples), n_traintest_samples)
    n_test = int(len(traintest_indices) * test_rate)

    val_indices = list(range(len(val_song_path_list)))
    test_indices = traintest_indices[:n_test]
    train_indices = traintest_indices[n_test:]

    def save(name, song_path_list, oke_path_list, indices):
        csv_path = os.path.join(root_dir, '%s.csv' % name)

        with open(csv_path, 'w') as csv_fp:
            writer = csv.writer(csv_fp)

            for index in indices:
                song_path = song_path_list[index]
                oke_path = oke_path_list[index]

                song_relpath = os.path.relpath(song_path, start=root_dir)
                oke_relpath = os.path.relpath(oke_path, start=root_dir)

                writer.writerow([ str(index), song_relpath, oke_relpath ])

    save('val', val_song_path_list, val_oke_path_list, val_indices)
    save('test', traintest_song_path_list, traintest_oke_path_list, test_indices)
    save('train', traintest_song_path_list, traintest_oke_path_list, train_indices)
