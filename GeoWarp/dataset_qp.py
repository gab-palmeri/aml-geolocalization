
import torch

import util
import datasets_util


class DatasetQP(torch.nn.Module):
    def __init__(self, model, global_features_dim, geoloc_train_dataset, qp_threshold):
        """Dataset used to compute pairs of query-positive. These pairs are then
        used in the weakly supervised losses (consistency and features-wise).
        
        Parameters
        ----------
        model : nn.Module, used to compute the query-positive pairs.
        global_features_dim : int, dimension of the global features generated by the model.
        geoloc_train_dataset : dataset_geoloc.GeolocDataset, containing the queries and gallery images.
        threshold : float, only pairs with distance (in features space) below
            the given threshold will be taken into account.
        """
        
        super().__init__()
        # Compute predictions with the given model on the given dataset
        _, _, predictions, correct_bool_mat, distances = util.compute_features(geoloc_train_dataset, model, global_features_dim)
        
        num_preds = predictions.shape[1]
        real_positives = [[] for _ in range(geoloc_train_dataset.queries_num)]
        
        # In query_positive_distances saves the index of query, positive, and their distance
        # for each query-positive pair
        query_positive_distances = []
        for query_index in range(geoloc_train_dataset.queries_num):
            query_path = geoloc_train_dataset.queries_paths[query_index]
            for pred_index in range(num_preds):
                if correct_bool_mat[query_index, pred_index] == 1:
                    distance = distances[query_index, pred_index]
                    positive = predictions[query_index, pred_index]
                    positive_path = geoloc_train_dataset.database_paths[positive]
                    real_positives[query_index].append(positive_path)
                    query_positive_distances.append((query_path, positive_path, distance))
        # Filter away the query-positive pairs which are further than qp_threshold from each other
        self.query_positive_distances = [qpd for qpd in query_positive_distances if qpd[2] < qp_threshold]
    
    def __getitem__(self, index):
        query_path, positive_path, _ = self.query_positive_distances[index]
        query = datasets_util.open_image_and_apply_transform(query_path)
        positive = datasets_util.open_image_and_apply_transform(positive_path)
        return query, positive
    
    def __len__(self):
        return len(self.query_positive_distances)
