import torch
import typing as t
from torch.utils.data import Dataset


class CacheableDataset(Dataset):
    __cache: t.Tuple[torch.Tensor]

    def __init__(self, *args, data: Dataset, cache: bool = False):
        super().__init__(*args)

        if not cache:
            self.__cache = data
        else:
            self.__cache = (entry for entry in data)

    def __len__(self) -> int:
        return len(self.__cache)

    def __getitem__(self, index) -> torch.Tensor:
        return self.__cache[index]
