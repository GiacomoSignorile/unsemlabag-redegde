from .PhenoBench import PhenoBench 
from .RedEdgeDataset import RedEdgeDataModule 

def get_dataset(name, dataset_opts):
    if name == "PhenoBench":
        return PhenoBench(dataset_opts)
    elif name == "RedEdge":
        return RedEdgeDataModule(dataset_opts) 
    else:
        raise ValueError("Dataset class not found for {}".format(name))