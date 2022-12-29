
import os
import numpy as np
from glob import glob
from os.path import join
import torch.utils.data as data
from sklearn.neighbors import NearestNeighbors

import datasets_util

def get__class_id__group_id(utm_east, utm_north, heading, M, alpha, N, L):
        """Return class_id and group_id for a given point.
            The class_id is a triplet (tuple) of UTM_east, UTM_north and
            heading (e.g. (396520, 4983800,120)).
            The group_id represents the group to which the class belongs
            (e.g. (0, 1, 0)), and it is between (0, 0, 0) and (N, N, L).
        """
        rounded_utm_east = int(utm_east // M * M)  # Rounded to nearest lower multiple of M
        rounded_utm_north = int(utm_north // M * M)
        rounded_heading = int(heading // alpha * alpha)
        
        class_id = (rounded_utm_east, rounded_utm_north, rounded_heading)
        # group_id goes from (0, 0, 0) to (N, N, L)
        group_id = (rounded_utm_east % (M * N) // M,
                    rounded_utm_north % (M * N) // M,
                    rounded_heading % (alpha * L) // alpha)
        return class_id, group_id


class GeolocDataset(data.Dataset):
    def __init__(self, dataset_folder="/content/small", dataset_name="sf-xs", split="train", positive_dist_threshold=25):
        
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset_folder = join(dataset_folder, split) #small/train or small/test
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
        
        if split == "train":
            #### Read paths and UTM coordinates for all images.
            gallery_proportion = 0.8
            queries_prop = 1 - gallery_proportion
            images_per_class = 10

            images_paths = sorted(glob(self.dataset_folder, "/**/*.jpg"), recursive=True)
            class_id_path = {}
            for path in images_paths:
                utm_east = float(path.split("@")[1])
                utm_north = float(path.split("@")[2])
                heading = float(path.split("@")[9])
                class_id, _ = get__class_id__group_id(utm_east, utm_north, heading, M=10, alpha=30, N=5, L=2)
                if class_id not in class_id_path:
                    class_id_path[class_id] = []
                class_id_path[class_id].append(path)

            self.database_paths = []
            self.queries_paths = []

            for class_id, paths in class_id_path.items():
                self.database_paths += paths[:int(images_per_class * gallery_proportion)] # 0 to 7
                self.queries_paths += paths[int(images_per_class * gallery_proportion):] # 8 to 9
            
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
