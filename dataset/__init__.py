from .zjumocap import ZJUMoCapDataset
from .people_snapshot import PeopleSnapshotDataset
from .x_humans import X_HumansDataset
from .fd_dress import FDressDataset
from .fd_dress_multiseq import FDressMultiseqDataset
from .fd_dress_multicam import FDressMulticamDataset
def load_dataset(cfg, split='train'):
    dataset_dict = {
        'zjumocap': ZJUMoCapDataset,
        'people_snapshot': PeopleSnapshotDataset,
        'x_humans': X_HumansDataset,
        '4Dress': FDressDataset,
        '4DressMultiseq': FDressMultiseqDataset,
        '4DressMulticam': FDressMulticamDataset,
    }
    return dataset_dict[cfg.name](cfg, split)
