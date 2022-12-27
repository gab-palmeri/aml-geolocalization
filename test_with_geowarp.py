
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
import GeoWarp.network as network
import GeoWarp.util as util
import GeoWarp.dataset_warp as dataset_warp

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


def test(args: Namespace, eval_ds: Dataset, model: torch.nn.Module) -> Tuple[np.ndarray, str]:
    """Compute descriptors of the given dataset and compute the recalls."""
    
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
    
    ##NEW PART
    reranked_predictions = predictions.copy()
    test_dataset = eval_ds
    num_reranked_predictions = 5
    test_batch_size = 1
    for num_q in tqdm(range(test_dataset.queries_num), desc="Testing", ncols=100):
            dot_prods_wqp = np.zeros((num_reranked_predictions))
            query_path = test_dataset.queries_paths[num_q]
            for i1 in range(0, num_reranked_predictions, test_batch_size):
                batch_indexes = list(range(num_reranked_predictions))[i1:i1+test_batch_size]
                current_batch_size = len(batch_indexes)
                query = open_image_and_apply_transform(query_path)
                query_repeated_twice = torch.repeat_interleave(query.unsqueeze(0), current_batch_size, 0)
                
                preds = []
                for i in batch_indexes:
                    pred_path = test_dataset.database_paths[predictions[num_q, i]]
                    preds.append(open_image_and_apply_transform(pred_path))
                preds = torch.stack(preds)
                
                ## MODEL
                features_extractor = network.FeaturesExtractor("alexnet", "gem")
                #global_features_dim = commons_warp.get_output_dim(features_extractor, "GEM")
                
                state_dict = torch.load("/content/alexnet_gem.pth")
                features_extractor.load_state_dict(state_dict)
                del state_dict
                
                state_dict = torch.load("/content/homography_regression.torch")
                homography_regression = network.HomographyRegression(kernel_sizes=[7, 5, 5, 5, 5, 5], channels=[225, 128, 128, 64, 64, 64, 64], padding=1)
                homography_regression.load_state_dict(state_dict)
                del state_dict
                
                warp_model = network.Network(features_extractor, homography_regression).cuda().eval()
                warp_model = torch.nn.DataParallel(warp_model)
                ##END
                warped_pair = dataset_warp.compute_warping(warp_model, query_repeated_twice.cuda(), preds.cuda())
                q_features = warp_model("features_extractor", [warped_pair[0], "local"])
                p_features = warp_model("features_extractor", [warped_pair[1], "local"])
                # Sum along all axes except for B. wqp stands for warped query-prediction
                dot_prod_wqp = (q_features * p_features).sum(list(range(1, len(p_features.shape)))).cpu().detach().numpy()
                
                dot_prods_wqp[i1:i1+test_batch_size] = dot_prod_wqp
            
            reranking_indexes = dot_prods_wqp.argsort()[::-1]
            reranked_predictions[num_q, :num_reranked_predictions] = predictions[num_q][reranking_indexes]
    
    ground_truths = test_dataset.get_positives()
    recalls, recalls_pretty_str = util.compute_recalls(reranked_predictions, ground_truths, test_dataset, RECALL_VALUES)
    return recalls, recalls_pretty_str

