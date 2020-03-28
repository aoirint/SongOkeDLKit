
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as LR
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

import BiAudioTransform as BT
import BiAudioTransform.functional as BTF
from SongOkeCroppedDataset import *

if __name__ == '__main__':
    root_dir = os.path.expanduser('~/datasets/MaouSongOkeCroppedDataset')
    epochs = 300
    batch_size = 32
    lr = 1e-2
    cpu_workers = 4
    dump_interval = 5
    log_dir = 'outputs'
    device = torch.device('cuda:0')

    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    test_dir = os.path.join(log_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)

    def conv1d(in_channels, out_channels, k, p):
        return nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=p)

    def relu():
        return nn.ReLU()

    def sigmoid():
        return nn.Sigmoid()

    model = nn.Sequential(
        conv1d(2, 16, k=3, p=1),
        relu(),
        conv1d(16, 32, k=3, p=1),
        conv1d(32, 64, k=3, p=1),
        relu(),
        conv1d(64, 32, k=3, p=1),
        relu(),
        conv1d(32, 16, k=3, p=1),
        relu(),
        conv1d(16, 2, k=3, p=1),
        sigmoid(),
    )
    model = model.to(device)

    train_transform = BT.Compose([
        BT.ToTensor(),
    ])
    test_transform = BT.Compose([
        BT.ToTensor(),
    ])

    train_dataset = SongOkeCroppedDataset(root_dir=root_dir, list_file='train.csv', bi_transform=train_transform)
    test_dataset = SongOkeCroppedDataset(root_dir=root_dir, list_file='test.csv', bi_transform=test_transform)

    log_loss = []
    def plot_loss():
        plt.clf()
        plt.title('Loss (Epoch %d; %f)' % (epoch, log_loss[-1]))
        epoch_list = list(range(1, epoch+1))
        plt.plot(epoch_list, log_loss)

        plt.savefig(os.path.join(log_dir, 'loss.png'))

    def save_checkpoint():
        checkpoint = {
            'model': model.state_dict(),
            'loss': log_loss,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'epoch': epoch,
        }
        if epoch % dump_interval == 0:
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint-%d.ckpt' % epoch))
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint-latest.ckpt'))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_workers)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=cpu_workers)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = LR.MultiStepLR(optimizer, milestones=[ 100, 200, ], gamma=0.1)

    for epoch in trange(1, epochs+1):
        sum_loss = 0
        model.train()
        for batch_index, (song, oke) in tqdm(enumerate(train_loader), total=len(train_loader)):
            song = song.to(device)
            oke = oke.to(device)

            pred = model(song)

            loss = F.mse_loss(oke, pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value = loss.cpu().item()
            sum_loss += loss_value
        scheduler.step()

        loss_value = sum_loss / len(train_loader)
        log_loss.append(loss_value)
        plot_loss()
        save_checkpoint()

        model.eval()
        for batch_index, (song, _) in tqdm(enumerate(test_loader), total=len(test_loader)):
            song = song.to(device)
            with torch.no_grad():
                pred = model(song)

            for inbatch_index in range(pred.shape[0]):
                pred_tensor = pred[inbatch_index]
                pred_audio = BTF.to_pydub(pred_tensor)

                pred_path = os.path.join(test_dir, '%d_%d.wav' % (batch_index, inbatch_index))
                pred_audio.export(pred_path, format='wav')
