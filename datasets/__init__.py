# File: datasets/__init__.py

from .PhenoBench import PhenoBench # Your existing one
from .RedEdgeDataset import RedEdgeDataModule # Import the new one

def get_dataset(name, dataset_opts):
    if name == "PhenoBench":
        return PhenoBench(dataset_opts)
    elif name == "RedEdge": # Or "WeedMap" or whatever you named it
        return RedEdgeDataModule(dataset_opts) # Use the new module
    else:
        raise ValueError("Dataset class not found for {}".format(name))