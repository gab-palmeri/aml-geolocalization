import faiss
import torch
import logging
from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import Tuple
from argparse import Namespace
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# Compute R@1, R@5, R@10, R@20
RECALL_VALUES = [1, 5, 10, 20]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def open_image_and_apply_transform(image_path):
    """Given the path of an image, open the image, and return it as a normalized tensor.
    """
    
    pil_image = Image.open(image_path)
    tensor_image = transform(pil_image)
    return tensor_image


def test(args: Namespace, eval_ds: Dataset, model: torch.nn.Module, today: torch.nn.Module) -> Tuple[np.ndarray, str]:
    """Compute descriptors of the given dataset and compute the recalls."""

    toDayModel = today
    toDayModel.load_state_dict(torch.load('150_net_G3.pth'))
    toDayModel = toDayModel.eval()
    
    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))
        all_descriptors = np.empty((len(eval_ds), args.fc_output_dim), dtype="float32")
        for images, indices in tqdm(database_dataloader, ncols=100):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
        
        logging.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_infer_batch_size = 1
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device == "cuda"))
        for images, indices in tqdm(queries_dataloader, ncols=100):

            toDayModel.set_input(images)
            toDayModel.test()

            #get the image
            img_path = toDayModel.get_image_paths()

            #open the image
            images = open_image_and_apply_transform(img_path)

            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
    
    queries_descriptors = all_descriptors[eval_ds.database_num:]
    database_descriptors = all_descriptors[:eval_ds.database_num]
    
    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args.fc_output_dim)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors
    
    logging.debug("Calculating recalls")
    _, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))
    
    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    recalls = np.zeros(len(RECALL_VALUES))
    for query_index, preds in enumerate(predictions):
        for i, n in enumerate(RECALL_VALUES):
            if np.any(np.in1d(preds[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by queries_num and multiply by 100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(RECALL_VALUES, recalls)])
    return recalls, recalls_str


    # toDayModel = network.GeoLocalizationNet()
    # toDayModel.load_state_dict(torch.load('model_best.pth'))

    # # test
    # for i, data in enumerate(eval_ds):
        
    #     toDayModel.set_input(data)
    #     toDayModel.test()
    #     img_path = toDayModel.get_image_paths()
    #     print('process image... %s' % img_path)

