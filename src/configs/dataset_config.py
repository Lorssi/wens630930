import os

class dataset_config:

    ROOT_DIR = "."
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    INTERIM_DIR = os.path.join(DATA_DIR, "interim")
    FEATURE_DIR = os.path.join(INTERIM_DIR, "feature_store")