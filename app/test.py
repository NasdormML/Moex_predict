import os
import pickle

METADATA_PATH = os.path.join("history", "metadata", "training_metadata.pkl")


def load_training_metadata():
    try:
        with open(METADATA_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


print(load_training_metadata())
