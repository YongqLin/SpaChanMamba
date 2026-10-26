# datasets/__init__.py
from .aid_dataset import get_aid_dataloaders
from .nwpu_dataset import get_nwpu_dataloaders
from .ucmerced_dataset import get_ucmerced_dataloaders
__all__ = [
    'get_aid_dataloaders',
    'get_nwpu_dataloaders',
    'get_ucmerced_dataloaders',
]
