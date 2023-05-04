"""
Wrapper function to train generic tensorflow DNNs to be explained
"""
import os.path
import pickle

import keras_tuner as kt
from pathlib import Path
from box import Box



def save_tuner(tuner: kt.Hyperband, config: Box, dataset: str) -> None:
    """
    Function to pickle the hyperband tuner; used such that the tuner
    is only run on one seed and the best parameters are used for the
    other seeds. Can be used to pickle any object
    Args:
        tuner (kt.Hyperband): fitted hyperband tuner (completed hyperparameter search)
        config (Box): main config
        dataset (str): name of dataset

    Returns:
        None: saves the hyperband tuner object as a pickle file in ``config.dnn_path/dataset/tuner.pkl``
    """
    save_path = Path(config.dnn_path).joinpath(f"{dataset}/tuner.pkl")
    if not os.path.exists(save_path.parent):
        os.makedirs(save_path.parent)

    with open(save_path, "wb") as f:
        pickle.dump(obj=tuner, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    return None

def load_tuner(config: Box, dataset: str) -> kt.Hyperband:
    load_path = Path(config.dnn_path).joinpath(f"{dataset}/tuner.pkl")

    with open(load_path, "rb") as f:
        tuner = pickle.load(f)
    return tuner
