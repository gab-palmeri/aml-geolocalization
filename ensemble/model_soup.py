import torch
import logging
import numpy as np
from os import listdir, path
from ..model.ensemble import get_model_from_sd
from ..datasets.test_dataset import SamTestDataset
from ..test import test
from ..model.network import GeoLocalizationNet

def evaluate_individual_models(args, weights_path, datasets):
    results = {}
    models = [x for x in listdir(weights_path) if "pth" in x]
    for i, weights in enumerate(models):
        logging.debug(f"({i}) Loading weights {weights}")

        # Load the weights
        state_dict = torch.load(path.join(weights_path, weights))
        model = GeoLocalizationNet(args.backbone, args.fc_output_dim, args.pretrain)
        model.load_state_dict(state_dict)
        model = model.to(args.device)
        model.eval()
        with torch.no_grad():
            # load the dataset
            for dataset in datasets:
                queries_folder = "queries_v1" if "small" in dataset and "test" in dataset else "queries"

                test_ds = SamTestDataset(dataset, queries_folder=queries_folder, positive_dist_threshold=args.positive_dist_threshold)
                recall, recall_str = test(args, test_ds, model, test_method=args.test_method)
                # free memory
                del test_ds
                
                if results[weights] is None:
                    results[weights] = {dataset: recall}
                else:
                    results[weights][dataset] = recall
                
                logging.debug(f"({i}) {weights} {dataset} {recall_str}")
        
        del model
        del state_dict
    return results

def greedy_soup(args, base_model, weights_path, datasets_paths_map, val_ds_path, results, sort_by="sf_r1"):
    assert sort_by in [
                        "tn_r1", "tn_r5" \
                        "sf_r1", "sf_r5", \
                        "ts_r1", "ts_r5", \
                        "sf_val_r1", "sf_val_r5"
                    ]

    # sort the models by the best recall on a specific dataset
    sorted_models = []
    if sort_by == "tn_r1":
        sorted_models = sorted(results.items(), key=lambda x: x[1][datasets_paths_map["tokyo-night"]][0], reverse=True)
    elif sort_by == "tn_r5":
        sorted_models = sorted(results.items(), key=lambda x: x[1][datasets_paths_map["tokyo-night"]][1], reverse=True)
    elif sort_by == "ts_r1":
        sorted_models = sorted(results.items(), key=lambda x: x[1][datasets_paths_map["tokyo-xs"]][0], reverse=True)
    elif sort_by == "ts_r5":
        sorted_models = sorted(results.items(), key=lambda x: x[1][datasets_paths_map["tokyo-xs"]][1], reverse=True)
    elif sort_by == "sf_r1":
        sorted_models = sorted(results.items(), key=lambda x: x[1][datasets_paths_map["sf-xs"]][0], reverse=True)
    elif sort_by == "sf_r5":
        sorted_models = sorted(results.items(), key=lambda x: x[1][datasets_paths_map["sf-xs"]][1], reverse=True)
    elif sort_by == "sf_val_r1":
        sorted_models = sorted(results.items(), key=lambda x: x[1][datasets_paths_map["sf-val"]][0], reverse=True)
    elif sort_by == "sf_val_r5":
        sorted_models = sorted(results.items(), key=lambda x: x[1][datasets_paths_map["sf-val"]][1], reverse=True)

    greedy_soup_ingredients = [sorted_models[0][0]]
    greedy_soup_params = torch.load(path.join(weights_path, sorted_models[0][0]))

    best_recall_so_far = sorted_models[0][1]["sf-val"][0]
    val_ds = SamTestDataset(val_ds_path, queries_folder="queries_v1", positive_dist_threshold=args.positive_dist_threshold)

    for i, weights in enumerate(results):
        if i == 0: continue

        logging.debug("Testing greedy soup with {weights}")
        new_ingredients = torch.load(path.join(weights_path, weights))
        num_ingredients = len(greedy_soup_ingredients)
        potential_ingredients = {
            k: greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1.0)) + new_ingredients[k].clone() * (1.0 / (num_ingredients + 1.0)) \
                for k in new_ingredients
        } 

        # run model with the new ingredients
        model = GeoLocalizationNet(args.backbone, args.fc_output_dim, args.pretrain)
        model.load_state_dict(potential_ingredients)
        model = model.to(args.device)
        model.eval()
        recall_to_compare = 0.
        with torch.no_grad():
            recall, recall_str = test(args, val_ds, model, test_method=args.test_method)
            recall_to_compare = recall[0]
            # free memory
            del model
            del potential_ingredients
            del recall
            del recall_str
        
        # if the new ingredients improve the recall, add them to the soup
        if recall_to_compare > best_recall_so_far:
            greedy_soup_ingredients.append(weights)
            best_recall_so_far = recall_to_compare
            greedy_soup_params = potential_ingredients
            logging.debug(f"New best recall: {best_recall_so_far}, adding to soup: {weights}")
    
    return greedy_soup_params, greedy_soup_ingredients


def evaluate_greedy_soup(args, base_model, datasets, params):
    results = {}
    
    model = GeoLocalizationNet(args.backbone, args.fc_output_dim, args.pretrain)
    model.load_state_dict(params)
    model = model.to(args.device)
    model.eval()
    with torch.no_grad():
            # load the dataset
            for dataset in datasets:
                queries_folder = "queries_v1" if "small" in dataset and "test" in dataset else "queries"

                test_ds = SamTestDataset(dataset, queries_folder=queries_folder, positive_dist_threshold=args.positive_dist_threshold)
                recall, recall_str = test(args, test_ds, model, test_method=args.test_method)
                # free memory
                del test_ds
                
                results[dataset] = recall
                
                logging.info(f"{dataset} {recall_str}")
    return results



