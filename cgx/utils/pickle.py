import pickle
from pathlib import Path
from typing import List

def write_object(object_, path: str) -> None:
    with open(Path(path), "wb") as f:
        pickle.dump(object_, f, pickle.HIGHEST_PROTOCOL)
    return None
def load_object(path: str) -> object:
    with open (Path(path), "rb") as f:
        object_ = pickle.load(f)
    return object_

def flatten(l) -> List:
    return [item for sublist in l for item in sublist]