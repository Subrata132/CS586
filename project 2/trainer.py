import os
import torch
import random
import argparse
import numpy as np
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from data_loader import LoadData
from model import CNNModel
from csv_to_numpy import create_folder


class Trainer:
    def __init__(
            self,
            batch_size=16,
            epochs=5,
            lr=1e-3
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.best_val_epoch = None
        self.best_val_loss = 1e8
        self.save_dir = "./saved_models"
        create_folder(self.save_dir)

    def train(self):
        all_filenames = os.listdir("../data/project_2")
        random.shuffle(all_filenames)
        train_split = int(0.7 * len(all_filenames))
        val_split = int(0.8 * len(all_filenames))
        train_filenames = all_filenames[:train_split]
        val_filenames = all_filenames[train_split:val_split]
        test_filenames = all_filenames[val_split:]
        train_loader = LoadData(batch_size=self.batch_size, filenames=train_filenames)
        val_loader = LoadData(batch_size=self.batch_size, filenames=val_filenames)
        test_loader = LoadData(batch_size=self.batch_size, filenames=test_filenames)

        model_name = "cnn_model.h5"
        model = CNNModel(in_channel=1).to(device=self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)

        for epoch in range(self.epochs):
            model.train()
            train_loss = 0
            count = 0
            progress_bar = tqdm(train_loader.load_data())
            progress_bar.set_description(f'Epoch: {epoch + 1}')
            for x, y in progress_bar:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                count += 1
                progress_bar.set_postfix({"loss": round(train_loss / count, 4)})
            print("\n")
            val_progress_bar = tqdm(val_loader.load_data())
            val_loss = 0
            count = 0
            val_y = []
            pred_val_y = []
            val_progress_bar.set_description(f'Validating: ')
            model.eval()
            for x, y in val_progress_bar:
                x, y = x.to(self.device), y.to(self.device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss += loss.item()
                count += 1
                val_progress_bar.set_postfix({"loss": round(val_loss / count, 4)})
                val_y = val_y + list(y.detach().cpu().numpy())
                y_pred = np.argmax(y_pred.detach().cpu().numpy(), axis=-1)
                pred_val_y = pred_val_y + list(y_pred)
            val_loss /= count
            accuracy = accuracy_score(y_true=val_y, y_pred=pred_val_y)
            print(f'Validation accuracy: {accuracy}')
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(self.save_dir, model_name))
                print(f'Model saved at epoch; {epoch + 1}\n')
        print(f'\nBest model saved at epoch {self.best_val_epoch} with validation loss: {self.best_val_loss}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=2, type=int)
    args = parser.parse_args()
    trainer = Trainer(
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    trainer.train()


if __name__ == '__main__':
    main()

