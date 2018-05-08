import os

class GlobalVars:

    project_root = os.path.dirname(os.path.abspath(__file__))
    batch_size = 1000
    raw_csv_path = os.path.join(project_root, "data", "Seeing-Machines-Hackathon-Dataset.csv")
    train_path = os.path.join(project_root, "data", "train.csv")
    val_path = os.path.join(project_root, "data", "validation.csv")
    emb_dim = 4
    model_dir = os.path.join(project_root, "checkpoint")

