import os
import torch
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
    
        #Concatenate database and queries path, prefix database with database- and queries with query-

        self.images_path = ["database-" + p for p in self.database_paths] + ["query-" + p for p in self.queries_paths]

        self.fourier_images = [
            open_image("/content/Team/FDA/images/1.jpg"),
            open_image("/content/Team/FDA/images/2.jpg"),
            open_image("/content/Team/FDA/images/3.jpg"),
            open_image("/content/Team/FDA/images/4.jpg"),
            open_image("/content/Team/FDA/images/5.jpg"),
            open_image("/content/Team/FDA/images/6.jpg"),
            open_image("/content/Team/FDA/images/7.jpg"),
            open_image("/content/Team/FDA/images/8.jpg"),
            open_image("/content/Team/FDA/images/9.jpg"),
            open_image("/content/Team/FDA/images/10.jpg")
            ]
        # self.images_paths = [p for p in self.database_paths]
        # self.images_paths += [p for p in self.queries_paths]
        
        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)
    
    def __getitem__(self, index):
        image_path = self.images_paths[index]

        if image_path.startswith("database-"):
            image_path = image_path.replace("database-", "")
            image = open_image(image_path)
            image = self.database_transform(image)
            return image, image_path, None
        elif image_path.startswith("query-"):
            image_path = image_path.replace("query-", "")
            im_src = open_image(image_path)
            im_size = np.asarray(im_src).shape

            #generate number between 1 and 10 and open a file with that name
            random_number = random.randint(1, 10)
            im_trg = self.fouried_images[random_number]

            im_src_resized = im_src.resize( (1024,512), Image.BICUBIC )
            im_trg_resized = im_trg.resize( (1024,512), Image.BICUBIC )

            im_src_arr = np.asarray(im_src_resized, np.float32)
            im_trg_arr = np.asarray(im_trg_resized, np.float32)

            im_src_arr_tps = im_src_arr.transpose((2, 0, 1))
            im_trg_arr_tps = im_trg_arr.transpose((2, 0, 1))

            src_in_trg = FDA_source_to_target_np(im_src_arr_tps, im_trg_arr_tps, L=0.01)
            src_in_trg = src_in_trg.transpose((1,2,0))

            final_image = Image.fromarray(scale(src_in_trg)).resize((im_size[1],im_size[0]), Image.BICUBIC)

            normalized_img = self.database_transform(final_image)
            return normalized_img, index

    
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self):
        return f"< {self.dataset_name} - #q: {self.queries_num}; #db: {self.database_num} >"
    
    def get_positives(self):
        return self.positives_per_query

class SamTestDataset(data.Dataset):
    def __init__(self, args, dataset_folder, database_folder="database",
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

        # data augmentation stuff
        self.test_method = args.test_method
        
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(f"Folder {self.dataset_folder} does not exist")
        if not os.path.exists(self.database_folder):
            raise FileNotFoundError(f"Folder {self.database_folder} does not exist")
        if not os.path.exists(self.queries_folder):
            raise FileNotFoundError(f"Folder {self.queries_folder} does not exist")
        
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        #### Read paths and UTM coordinates for all images.
        self.database_paths = sorted(glob(os.path.join(self.database_folder, "**", "*.jpg"), recursive=True))
        self.queries_paths = sorted(glob(os.path.join(self.queries_folder, "**", "*.jpg"),  recursive=True))
        
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(float)
        self.queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(float)
        
        # set db image size (height, width)
        self.db_image_size = Image.open(self.database_paths[0]).size if args.resize_as_db else None
        self.db_image_size = (self.db_image_size[1], self.db_image_size[0]) if self.db_image_size else None

        # Find positives_per_query, which are within positive_dist_threshold (default 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.positives_per_query = knn.radius_neighbors(self.queries_utms,
                                                        radius=positive_dist_threshold,
                                                        return_distance=False)
        
        self.images_paths = [p for p in self.database_paths]
        self.images_paths += [p for p in self.queries_paths]
        
        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)
    
    def __getitem__(self, index):
        image_path = self.images_paths[index]
        pil_img = open_image(image_path)
        normalized_img = self.base_transform(pil_img)

        if self.db_image_size:
            if self.test_method == "database":
                normalized_img = transforms.functional.resize(normalized_img, self.db_image_size)
            else:
                normalized_img = self._test_query_transform(normalized_img)

        return normalized_img, index
    
    def _test_query_transform(self, img):
        """Transform query image according to self.test_method."""
        C, H, W = img.shape
        if self.test_method == "central_crop":
            scale = max(self.db_image_size[0]/H, self.db_image_size[1]/W)
            proc_img = torch.nn.functional.interpolate(img.unsqueeze(0), scale_factor=scale).squeeze(0)
            proc_img = transforms.functional.center_crop(proc_img, self.db_image_size)
            assert proc_img.shape[1:] == torch.Size(self.db_image_size), f"proc_img.shape: {proc_img.shape}, db_image_size: {self.db_image_size}"
        elif self.test_method == "five_crops" or self.test_method == 'nearest_crop' or self.test_method == 'maj_voting':
            # Get 5 square crops with size==shorter_side (usually 480). Preserves ratio and allows batches.
            shorter_side = min(self.db_image_size)
            proc_img = transforms.functional.resize(img, shorter_side)
            proc_img = torch.stack(transforms.functional.five_crop(proc_img, shorter_side))
            assert proc_img.shape == torch.Size([5, 3, shorter_side, shorter_side]), \
                f"{proc_img.shape} {torch.Size([5, 3, shorter_side, shorter_side])}"
        else:
            # single query
            proc_img = transforms.functional.resize(img, min(self.db_image_size))
        return proc_img


    
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self):
        return f"< {self.dataset_name} - #q: {self.queries_num}; #db: {self.database_num} >"
    
    def get_positives(self):
        return self.positives_per_query
