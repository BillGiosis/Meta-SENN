import numpy as np

from dataset_classes.cub200 import CUB200Class

def get_cub_alignment_from_features(features_train_sorted):
    metric_matrix = compute_metric_matrix(np.array(features_train_sorted), "train")
    return np.mean(np.max(metric_matrix, axis=1))

def compute_metric_matrix(features, mode):
    image_attribute_labels = CUB200Class.get_image_attribute_labels(train=mode == "train")
    image_attribute_labels = CUB200Class.filter_attribute_labels(image_attribute_labels)
    matrix_shape = (features.shape[1], max(image_attribute_labels["attribute"]) + 1)
    accuracy_matrix = np.zeros(matrix_shape)
    sensitivity_matrix = np.zeros_like(accuracy_matrix)
    grouped_attributes = image_attribute_labels.groupby("attribute")
    for attribute_id, group in grouped_attributes:
        is_present = group[group["is_present"]]
        not_present = group[~group["is_present"]]
        
        # Filter out invalid indices
        valid_is_present_indices = is_present["img_id"][is_present["img_id"] < features.shape[0]]
        valid_not_present_indices = not_present["img_id"][not_present["img_id"] < features.shape[0]]
        
        if len(valid_is_present_indices) > 0:
            is_present_avg = np.mean(features[valid_is_present_indices], axis=0)
        else:
            is_present_avg = np.zeros(features.shape[1])
        
        if len(valid_not_present_indices) > 0:
            not_present_avg = np.mean(features[valid_not_present_indices], axis=0)
        else:
            not_present_avg = np.zeros(features.shape[1])
        
        sensitivity_matrix[:, attribute_id] = not_present_avg
        accuracy_matrix[:, attribute_id] = is_present_avg
    
    metric_matrix = accuracy_matrix - sensitivity_matrix
    no_abs_features = features - np.min(features, axis=0)
    no_abs_feature_mean = metric_matrix / no_abs_features.mean(axis=0)[:, None]
    return no_abs_feature_mean