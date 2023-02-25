import os
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader


class CustomDataloader(Dataset):
    def __init__(
            self,
            filenames,
            data_dir="../data/project_2"
    ):
        self.data_dir = data_dir
        self.filenames = filenames
        random.shuffle(self.filenames)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        x = np.load(os.path.join(self.data_dir, filename))[0]
        x = x.astype('float32')
        x = x.reshape((1, x.shape[0], x.shape[1]))
        y = int(filename.split("_")[3])
        return x, y


class LoadData:
    def __init__(self, batch_size, filenames):
        self.batch_size = batch_size
        self.filenames = filenames

    def load_data(self):
        dataset = CustomDataloader(filenames=self.filenames)
        data_loader = DataLoader(
            dataset=dataset, batch_size=self.batch_size
        )
        return data_loader


def main():
    all_filenames = os.listdir("../data/project_2")
    random.shuffle(all_filenames)
    train_split = int(0.7*len(all_filenames))
    val_split = int(0.8*len(all_filenames))
    train_filenames = all_filenames[:train_split]
    val_filenames = all_filenames[train_split:val_split]
    test_filenames = all_filenames[val_split:]
    data_loader = LoadData(batch_size=16, filenames=train_filenames)
    for i, (x, y) in enumerate(data_loader.load_data()):
        print(f'{i}: X shape: {x.shape} | y shape: {y.shape}')


if __name__ == '__main__':
    main()