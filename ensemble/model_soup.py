import torch
import logging
import numpy as np
from os import listdir, path
from ..datasets.test_dataset import DataAugTestDataset
from ..test import test
from ..model.network import GeoLocalizationNet

# ModelSoup: https://github.com/mlfoundations/model-soups

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
                queries_folder = "queries_v1" if "small" in datasets[dataset] and "test" in datasets[dataset] else "queries"
                logging.debug(f"Testing dataset {dataset}")
                test_ds = DataAugTestDataset(datasets[dataset], queries_folder=queries_folder, positive_dist_threshold=args.positive_dist_threshold, test_method=args.test_method, resize = args.resize_as_db)
                recall, recall_str = test(args, test_ds, model, test_method=args.test_method)
                # free memory
                del test_ds
                
                if weights in results:
                    results[weights][dataset] = recall
                else:
                    results[weights] = {dataset: recall}
                
                logging.info(f"({i}) {weights} {dataset} {recall_str}")
        
        del model
        del state_dict
    return results

def greedy_soup(args, weights_path, datasets_paths_map, results, sort_by="sf_r1"):
    assert sort_by in [
                        "tn_r1", "tn_r5", \
                        "sf_r1", "sf_r5", \
                        "ts_r1", "ts_r5", \
                        "sf_val_r1", "sf_val_r5"
                    ], f"sort_by must be one of the following: tn_r1, tn_r5, sf_r1, sf_r5, ts_r1, ts_r5, sf_val_r1, sf_val_r5. Got {sort_by}"

    # sort the models by the best recall on a specific dataset
    sorted_models = []
    if sort_by == "tn_r1":
        sorted_models = sorted(results.items(), key=lambda x: x[1]["tokyo-night"][0], reverse=True)
    elif sort_by == "tn_r5":
        sorted_models = sorted(results.items(), key=lambda x: x[1]["tokyo-night"][1], reverse=True)
    elif sort_by == "ts_r1":
        sorted_models = sorted(results.items(), key=lambda x: x[1]["tokyo-xs"][0], reverse=True)
    elif sort_by == "ts_r5":
        sorted_models = sorted(results.items(), key=lambda x: x[1]["tokyo-xs"][1], reverse=True)
    elif sort_by == "sf_r1":
        sorted_models = sorted(results.items(), key=lambda x: x[1]["sf-xs"][0], reverse=True)
    elif sort_by == "sf_r5":
        sorted_models = sorted(results.items(), key=lambda x: x[1]["sf-xs"][1], reverse=True)
    elif sort_by == "sf_val_r1":
        sorted_models = sorted(results.items(), key=lambda x: x[1]["sf-val"][0], reverse=True)
    elif sort_by == "sf_val_r5":
        sorted_models = sorted(results.items(), key=lambda x: x[1]["sf-val"][1], reverse=True)


    greedy_soup_ingredients = [sorted_models[0][0]]
    greedy_soup_params = torch.load(path.join(weights_path, sorted_models[0][0]))

    queries = "queries"
    if "tn" in sort_by:
        key = "tokyo-night"
    elif "ts" in sort_by:
        key = "tokyo-xs"
    elif "sf_val" in sort_by:
        key = "sf-val"
    else:
        key = "sf-xs"
        queries = "queries_v1"
    
    best_recall_so_far = sorted_models[0][1][key][0]
    val_ds = DataAugTestDataset(datasets_paths_map[key], queries_folder=queries, positive_dist_threshold=args.positive_dist_threshold, test_method=args.test_method, resize = args.resize_as_db)

    for i, weights in enumerate([x[0] for x in sorted_models]):
        if i == 0: continue

        logging.debug(f"Testing greedy soup with {weights}")
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
            logging.info(f"Recall: {recall}")
            # free memory
            del model
            del recall
            del recall_str
        
        # if the new ingredients improve the recall, add them to the soup
        if recall_to_compare > best_recall_so_far:
            args.add_to_greedy_soup = True
            greedy_soup_ingredients.append(weights)
            best_recall_so_far = recall_to_compare
            greedy_soup_params = potential_ingredients
            logging.debug(f"New best recall: {best_recall_so_far}, adding to soup: {weights}")
        else:
            logging.debug(f"New ingredients don't improve the recall... don't add them to the soup")
        del potential_ingredients
    
    return greedy_soup_params, greedy_soup_ingredients


def evaluate_greedy_soup(args, datasets, params):
    results = {}
    
    model = GeoLocalizationNet(args.backbone, args.fc_output_dim, args.pretrain)
    model.load_state_dict(params)
    model = model.to(args.device)
    model.eval()
    with torch.no_grad():
            # load the dataset
            for k in datasets:
                queries_folder = "queries_v1" if "small" in datasets[k] and "test" in datasets[k] else "queries"

                test_ds = DataAugTestDataset(datasets[k], queries_folder=queries_folder, positive_dist_threshold=args.positive_dist_threshold, test_method=args.test_method, resize = args.resize_as_db)
                recall, recall_str = test(args, test_ds, model, test_method=args.test_method)
                # free memory
                del test_ds
                
                results[k] = recall
                
                logging.info(f"{k} {recall_str}")
    return results



