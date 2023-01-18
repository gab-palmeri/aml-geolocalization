
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from typing import Tuple
from argparse import Namespace
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader, Dataset


# Compute R@1, R@5, R@10, R@20
RECALL_VALUES = [1, 5, 10, 20]


def test(args: Namespace, eval_ds: Dataset, model: torch.nn.Module, test_method = "database") -> Tuple[np.ndarray, str]:
    """Compute descriptors of the given dataset and compute the recalls."""
    
    if test_method:
        assert test_method in ["database", "single_query", "central_crop", "five_crops",
                           "nearest_crop", "maj_voting", "five_custom"], f"test_method can't be {test_method}"

    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database descriptors for evaluation/testing")
        eval_ds.test_method = "database"
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))

        # consider the case in which data augmentation is used
        if test_method == "nearest_crop" or test_method == "maj_voting" or test_method == "five_custom":
            all_descriptors = np.empty((5*eval_ds.queries_num + eval_ds.database_num, args.fc_output_dim), dtype="float32")
        else:
            all_descriptors = np.empty((len(eval_ds), args.fc_output_dim), dtype="float32")

        for images, indices in tqdm(database_dataloader, ncols=100):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors
        
        queries_infer_batch_size = 1 if test_method == "single_query" else args.infer_batch_size
        logging.debug(f"Extracting queries descriptors for evaluation/testing using batch size {queries_infer_batch_size}")
        
        eval_ds.test_method = test_method
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device == "cuda"))
        for images, indices in tqdm(queries_dataloader, ncols=100):
            if test_method in ["five_crops", "nearest_crop", "maj_voting", "five_custom"]:
                images = torch.cat(tuple(images))
            descriptors = model(images.to(args.device))
            if test_method == "five_crops":
                descriptors = torch.stack(torch.split(descriptors, 5)).mean(1)
            descriptors = descriptors.cpu().numpy()

            if test_method in ["nearest_crop", "maj_voting", "five_custom"]:
                start_idx = eval_ds.database_num + 5*(indices[0] - eval_ds.database_num)
                end_idx = start_idx + 5 * indices.shape[0]
                indices = np.arange(start_idx, end_idx)
                all_descriptors[indices, :] = descriptors
            else:
                all_descriptors[indices.numpy(), :] = descriptors
    
    queries_descriptors = all_descriptors[eval_ds.database_num:]
    database_descriptors = all_descriptors[:eval_ds.database_num]
    
    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(args.fc_output_dim)
    faiss_index.add(database_descriptors)
    del database_descriptors, all_descriptors
    
    logging.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_descriptors, max(RECALL_VALUES))

    # post processing

    if test_method in ["nearest_crop", "five_custom"]:
        distances = np.reshape(distances, (eval_ds.queries_num, 20*5))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 20*5))
        for q in range(eval_ds.queries_num):
            # sort predictions by distance 
            sort_idx = np.argsort(distances[q])
            predictions[q] = predictions[q, sort_idx]
            # remove duplicated
            _, unique_idx = np.unique(predictions[q], return_index=True)
            # sort again
            predictions[q, :20] = predictions[q, np.sort(unique_idx)][:20]
        predictions = predictions[:, :20] # keep only the first 20 predictions
    elif test_method == "maj_voting":
        distances = np.reshape(distances, (eval_ds.queries_num, 5, 20))
        predictions = np.reshape(predictions, (eval_ds.queries_num, 5, 20))
        for q in range(eval_ds.queries_num):
            # votings, modify distance in place
            top_n_voting('top1', predictions[q], distances[q], args.maj_weight)
            top_n_voting('top5', predictions[q], distances[q], args.maj_weight)
            top_n_voting('top10', predictions[q], distances[q], args.maj_weight)

            # flatten dist
            dist = distances[q].flatten()
            preds = predictions[q].flatten()

            # sort predictions by distance
            sort_idx = np.argsort(dist)
            preds = preds[sort_idx]
            # remove duplicated
            _, unique_idx = np.unique(preds, return_index=True)
            # sort again
            predictions[q, 0, :20] = preds[np.sort(unique_idx)][:20]
        predictions = predictions[:, 0, :20] # keep only the first 20 predictions for each query
    
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


def top_n_voting(topn, predictions, distances, maj_weight):
    if topn == 'top1':
        n = 1
        selected = 0
    elif topn == 'top5':
        n = 5
        selected = slice(0, 5)
    elif topn == 'top10':
        n = 10
        selected = slice(0, 10)
    # find predictions that repeat in the first, first five,
    # or fist ten columns for each crop
    vals, counts = np.unique(predictions[:, selected], return_counts=True)
    # for each prediction that repeats more than once,
    # subtract from its score
    for val, count in zip(vals[counts > 1], counts[counts > 1]):
        mask = (predictions[:, selected] == val)
        distances[:, selected][mask] -= maj_weight * count/n
