import os
import random
import numpy as np
from glob import glob
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from sklearn.neighbors import NearestNeighbors

from FDA.fda import FDA_source_to_target_np, scale


def open_image(path):
    return Image.open(path).convert("RGB")


class TestDataset(data.Dataset):
    def __init__(self, dataset_folder, database_folder="database",
                 queries_folder="queries", positive_dist_threshold=25):
        """Dataset with images from database and queries, used for validation and test.
        Parameters
        ----------
        dataset_folder : str, should contain the path to the val or test set,
            which contains the folders {database_folder} and {queries_folder}.
        database_folder : str, name of folder with the database.
        queries_folder : str, name of folder with the queries.
        positive_dist_threshold : int, distance in meters for a prediction to
            be considered a positive.
        """
        super().__init__()
        self.dataset_folder = dataset_folder
        self.database_folder = os.path.join(dataset_folder, database_folder)
        self.queries_folder = os.path.join(dataset_folder, queries_folder)
        self.dataset_name = os.path.basename(dataset_folder)
        
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
        if not os.path.exists(self.database_folder):
            raise FileNotFoundError(f"Folder {self.database_folder} does not exist")
        if not os.path.exists(self.queries_folder):
            raise FileNotFoundError(f"Folder {self.queries_folder} does not exist")
        
        #DATABASE TRANSFORM
        self.database_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        #QUERY TRANSFORM -> WITH FOURIER DATA AUGMENTATION
        # self.queries_transform = transforms.Compose([
        #     FourierAugmentation().__call__(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])
        
        #### Read paths and UTM coordinates for all images.
        self.database_paths = sorted(glob(os.path.join(self.database_folder, "**", "*.jpg"), recursive=True))
        self.queries_paths = sorted(glob(os.path.join(self.queries_folder, "**", "*.jpg"),  recursive=True))
        
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(float)
        self.queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(float)
        
        # Find positives_per_query, which are within positive_dist_threshold (default 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.positives_per_query = knn.radius_neighbors(self.queries_utms,
                                                        radius=positive_dist_threshold,
                                                        return_distance=False)
        
        # for each string in database_paths, create a string which contains that string preceded by -database
        # for each string in queries_paths, create a string which contains that string preceded by -query
        # concatenate the two lists

        # self.images_paths = ["database-" + p for p in self.database_paths] + ["query-" + p for p in self.queries_paths]
        


        self.images_paths = [p for p in self.database_paths]
        self.images_paths += [p for p in self.queries_paths]
        
        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)
    
    def __getitem__(self, index):
        image_path = self.images_paths[index]
        pil_img = open_image(image_path)

        #generate number between 1 and 10 and open a file with that name
        random_number = random.randint(1, 10)
        pil_img2 = open_image(str(random_number) + ".jpg")

        im_src_resized = pil_img.resize( (1024,512), Image.BICUBIC )
        im_trg_resized = pil_img2.resize( (1024,512), Image.BICUBIC )

        im_src_arr = np.asarray(im_src_resized, np.float32)
        im_trg_arr = np.asarray(im_trg_resized, np.float32)

        im_src_arr_tps = im_src_arr.transpose((2, 0, 1))
        im_trg_arr_tps = im_trg_arr.transpose((2, 0, 1))

        #apply FDA from between pil_img and pil_img2
        pil_img = FDA_source_to_target_np(im_src_arr_tps, im_trg_arr_tps, random_number)

        pil_img = scale(pil_img.transpose((1,2,0)))

        normalized_img = self.database_transform(pil_img)
        return normalized_img, index

    
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self):
        return f"< {self.dataset_name} - #q: {self.queries_num}; #db: {self.database_num} >"
    
    def get_positives(self):
        return self.positives_per_query