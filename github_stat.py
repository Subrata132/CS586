import os
import requests
import json
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

random.seed(42)


class GitStats:
    def __init__(self, bin_size, credentials, max_github_id=123594509):
        self.bin_size = bin_size
        self.credentials = credentials
        self.max_github_id = max_github_id
        self.all_bins = list(range(0, self.max_github_id, self.bin_size))
        self.save_dir = "./saved_json/"

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
        json_files = os.listdir(self.save_dir)
        active_pcts = []
        for json_file in json_files:
            with open(os.path.join(self.save_dir, json_file)) as file:
                data = json.load(file)
            file.close()
            for key, value in data.items():
                active_pcts = active_pcts + value
        mean = np.mean(active_pcts)
        std = np.std(active_pcts)
        print(f'# Active Github users: {int(mean*self.max_github_id)} with standard deviation of {round(std, 4)}')

    def plot_result(self):
        json_files = sorted(os.listdir(self.save_dir))
        print(json_files)
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
        plt.show()


def main():
    token = "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"
    headers = {
        'Authorization': f'token {token}'
    }
    git_stat = GitStats(bin_size=1000, credentials=headers)
    # for select_bin in range(10, 51, 10):
    #     git_stat.crawl_data(select_bin=select_bin)
    git_stat.estimate_active_account()
    git_stat.plot_result()


if __name__ == '__main__':
    main()
