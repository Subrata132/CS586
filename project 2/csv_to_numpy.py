import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


class LatLong2Grid:
    def __init__(self):
        self.min_lat = 22.502
        self.max_lat = 22.688
        self.min_long = 113.813
        self.max_long = 114.293
        self.max_grid_long = 48
        self.max_grid_lat = 18
        self.plates = [0, 1, 2, 3, 4]
        self.data_dir = "../data/data_5drivers"
        self.save_dir = "../data/project_2/"
        create_folder(self.save_dir)

    def process_df(self, df):
        df['time'] = pd.to_datetime(df["time"])
        df.sort_values(by="time")
        df['longitude'] = df['longitude'].round(3)
        df['latitude'] = df['latitude'].round(3)
        s1 = ((df['longitude'] <= self.max_long) & (df['longitude'] >= self.min_long))
        s2 = ((df['latitude'] <= self.max_lat) & (df['latitude'] >= self.min_lat))
        slice_ = s1 & s2
        df = df[slice_]
        df['long'] = df['longitude'].apply(lambda x: int(
            self.max_grid_long * ((x - self.min_long) / (self.max_long - self.min_long)))
                                           )
        df['lat'] = df['latitude'].apply(lambda x: int(
            self.max_grid_lat - self.max_grid_lat * ((x - self.min_lat) / (self.max_lat - self.min_lat)))
                                         )
        return df

    def get_indicies(self, df):
        df = df.reset_index()
        df['flag'] = df['status'].diff(periods=1)
        start = df.iloc[0].status
        df.loc[0, 'flag'] = 0
        indicies = df[df['flag'].isin([-1, 1])].index.to_list()
        indicies.insert(0, 0)
        indicies.append(df.shape[0])
        return indicies, start

    def draw_trajectory(self, df):
        color = (255, 255, 255)
        thickness = 2
        image = np.zeros((self.max_grid_lat, self.max_grid_long), np.uint8) + .1
        points = list(df[['long', 'lat']].to_numpy())
        points = np.asarray(points, np.int32)
        image = cv2.polylines(image, [points], False, color, thickness)
        return image

    def get_feature(self, df):
        indices, start = self.get_indicies(df)
        all_features = []
        for j in range(0, len(indices) - 2, 2):
            if start == 1:
                first_image = self.draw_trajectory(df[indices[j]:indices[j + 1]])
                second_image = self.draw_trajectory(df[indices[j + 1]:indices[j + 2]])
            else:
                first_image = self.draw_trajectory(df[indices[j + 1]:indices[j + 2]])
                second_image = self.draw_trajectory(df[indices[j]:indices[j + 1]])
            all_features.append(np.stack([first_image, second_image], axis=0))
        return all_features

    def save_data(self):
        filenames = os.listdir(self.data_dir)
        for filename in tqdm(filenames):
            df = pd.read_csv(os.path.join(self.data_dir, filename))
            df = self.process_df(df)
            plates = df['plate'].unique()
            for plate in plates:
                plate_df = df[df['plate'] == plate]
                features = self.get_feature(df=plate_df)
                count = 0
                for feature in features:
                    npy_filename = filename.split(".")[0] + "_" + str(plate) + "_" + str(count) + ".npy"
                    count += 1
                    np.save(os.path.join(self.save_dir, npy_filename), feature)


def main():
    lat_long_to_grid = LatLong2Grid()
    lat_long_to_grid.save_data()


if __name__ == '__main__':
    main()
