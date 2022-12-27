
import os
import numpy as np
from glob import glob
from os.path import join
import torch.utils.data as data
from sklearn.neighbors import NearestNeighbors

import datasets_util


class GeolocDataset(data.Dataset):
    def __init__(self, dataset_folder="/content/small", dataset_name="sf-xs", split="train", positive_dist_threshold=25):
        
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_folder = join(dataset_folder, split) #small/train or small/test
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
        
        if split == "train":
            #### Read paths and UTM coordinates for all images.
            self.database_paths = sorted(glob(join(self.dataset_folder, "37.7*", "*.jpg"), recursive=True))
            self.queries_paths = sorted(glob(join(self.dataset_folder, "37.8*", "*.jpg"), recursive=True))
            
            # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
            self.database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(np.float)
            self.queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(np.float)
            
            # Find soft_positives_per_query, which are within positive_dist_threshold (deafult 25 meters)
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.database_utms)
            self.soft_positives_per_query = knn.radius_neighbors(self.queries_utms,
                                                                radius=positive_dist_threshold,
                                                                return_distance=False)
            self.images_paths = list(self.database_paths) + list(self.queries_paths)
            
            self.database_num = len(self.database_paths)
            self.queries_num = len(self.queries_paths)
        else:
            #### Read paths and UTM coordinates for all images.
            database_folder = join(self.dataset_folder, "database")
            queries_folder = join(self.dataset_folder, "queries_v1")
            if not os.path.exists(database_folder):
                raise FileNotFoundError(f"Folder {database_folder} does not exist")
            if not os.path.exists(queries_folder):
                raise FileNotFoundError(f"Folder {queries_folder} does not exist")
            self.database_paths = sorted(glob(join(database_folder, "*.jpg"), recursive=True))
            self.queries_paths = sorted(glob(join(queries_folder, "*.jpg"), recursive=True))
            
            # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
            self.database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(np.float)
            self.queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(np.float)
            
            # Find soft_positives_per_query, which are within positive_dist_threshold (deafult 25 meters)
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.database_utms)
            self.soft_positives_per_query = knn.radius_neighbors(self.queries_utms,
                                                                radius=positive_dist_threshold,
                                                                return_distance=False)
            self.images_paths = list(self.database_paths) + list(self.queries_paths)
            
            self.database_num = len(self.database_paths)
            self.queries_num = len(self.queries_paths)
    
    def __getitem__(self, index):
        image_path = self.images_paths[index]
        img = datasets_util.open_image_and_apply_transform(image_path)
        return img, index
    
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self):
        return f"< {self.__class__.__name__}, {self.dataset_name} - #database: {self.database_num}; #queries: {self.queries_num} >"
    
    def get_positives(self):
        return self.soft_positives_per_query
