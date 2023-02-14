import os
import requests
import json
import random
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
import matplotlib.pyplot as plt

random.seed(42)


class GitStats:
    def __init__(self, bin_size, credentials, max_github_id, max_validation_id, val_step=1000):
        self.bin_size = bin_size
        self.credentials = credentials
        self.max_github_id = max_github_id
        self.max_validation_id = max_validation_id
        self.val_step = val_step
        self.all_bins = list(range(0, self.max_github_id, self.bin_size))
        self.save_dir = "./saved_json/"
        self.active_percentages = []
        self.active_stds = []

    def crawl_data(self, select_bin):
        selected_bins = random.sample(self.all_bins, select_bin)
        active_percentage_dict = {}
        for bin_ in tqdm(selected_bins):
            id_ = bin_
            in_active = 0
            active_percentages = []
            while id_ < (bin_ + self.bin_size - 1):
                response = requests.get('https://api.github.com/users?since=' + str(id_), headers=self.credentials)
                data = response.json()
                in_active += ((data[-1]["id"]) - id_ - len(data))
                id_ = data[-1]["id"]
                active_percentages.append(1 - (in_active / self.bin_size))
            active_percentage_dict[bin_] = active_percentages
        with open(os.path.join(self.save_dir, f'active_percentage_{select_bin}.json'), "w") as outfile:
            json.dump(active_percentage_dict, outfile)
        outfile.close()

    def estimate_active_account(self):
        json_files = sorted(os.listdir(self.save_dir))
        table = PrettyTable(["# of bins", "mean", "std", "# of active account"])
        for json_file in json_files:
            active_pcts = []
            row = [json_file.split(".")[0].split("_")[-1]]
            with open(os.path.join(self.save_dir, json_file)) as file:
                data = json.load(file)
            file.close()
            for key, value in data.items():
                active_pcts = active_pcts + value
            mean = round(np.mean(active_pcts), 4)
            std = round(np.std(active_pcts), 4)
            row.append(mean)
            row.append(std)
            row.append(int(mean*self.max_validation_id))
            table.add_row(row)
            self.active_percentages.append(mean)
            self.active_stds.append(std)
        print("\t Result on entire ID space")
        print(table)
        print("\n")

    def validate_result(self, fold=5):
        active_pcts = []
        for _ in tqdm(range(fold)):
            current_id = random.randint(self.max_github_id+1, self.max_validation_id)
            end_ = current_id + self.val_step
            in_active_count = 0
            while current_id <= end_:
                response = requests.get(
                    'https://api.github.com/users?since=' + str(current_id),
                    headers=self.credentials
                )
                data = response.json()
                in_active_count += ((data[-1]["id"]) - current_id - len(data))
                current_id = data[-1]["id"]
            active_pcts.append(1 - in_active_count/self.val_step)
        ticks = list(range(1, len(active_pcts) + 1))
        plt.figure()
        plt.axhline(y=self.active_percentages[2], color='green', linewidth=2, label='Predicted active percentage')
        plt.plot(ticks, active_pcts, color='blue', linestyle='--', label='Actual active percentage')
        plt.xticks(ticks)
        plt.xlim(ticks[0], ticks[-1])
        plt.ylim(np.min(active_pcts) - 0.05, 1)
        plt.grid()
        plt.legend(loc='best')

    def plot_result(self):
        json_files = sorted(os.listdir(self.save_dir))
        fig = plt.figure(figsize=(5, 5))
        all_data = []
        label = []
        for json_file in json_files:
            with open(os.path.join(self.save_dir, json_file)) as file:
                data = json.load(file)
            file.close()
            active_pcts = []
            for key, value in data.items():
                active_pcts = active_pcts + value
            all_data.append(active_pcts)
            label.append(json_file.split(".")[0].split("_")[-1])
        plt.boxplot(all_data, patch_artist=True, labels=label)
        plt.grid()


def main():
    token = ""  # Add token here
    headers = {
        'Authorization': f'token {token}'
    }
    max_github_id = 63048878  # Till 2020-31-12
    max_validation_id = 123594509  # Till 2023-25-01
    git_stat = GitStats(
        bin_size=1000,
        credentials=headers,
        max_github_id=max_github_id,
        max_validation_id=max_validation_id
    )
    for select_bin in range(10, 51, 10):
        git_stat.crawl_data(select_bin=select_bin)
    git_stat.estimate_active_account()
    git_stat.plot_result()
    git_stat.validate_result()
    plt.show()


if __name__ == '__main__':
    main()
