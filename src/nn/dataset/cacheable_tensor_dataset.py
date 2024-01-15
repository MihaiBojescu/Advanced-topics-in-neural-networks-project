import os
import torch
import typing as t
from torch.utils.data import Dataset


class CacheableTensorDataset(Dataset):
    __cache: t.Tuple[torch.Tensor]

    def __init__(self, *args, dataset: Dataset, cache: bool = False, cache_path: t.Optional[str] = None):
        super().__init__(*args)

        if not cache:
            self.__cache = dataset
            return

        if self.__is_cache_available(dataset, cache_path):
            self.__cache = self.__load_cache(cache_path)
            return

        self.__cache = tuple(entry for entry in dataset)
        self.__save_cache(self.__cache, cache_path)

    def __is_cache_available(self, dataset: Dataset, cache_path: t.Optional[str]):
        if cache_path is None:
            return False

        if not os.path.exists(cache_path):
            return False

        return len(os.listdir(cache_path)) == len(dataset)

    def __load_cache(self, cache_path: str) -> t.Tuple[torch.Tensor]:
        cache = []
        files = os.listdir(cache_path)

        for file in files:
            tensor = torch.load(f"{cache_path}/{file}")
            cache.append((tensor, os.path.splitext(file)[0]))

        return tuple(cache)

    def __save_cache(self, cache: t.Tuple[torch.Tensor], cache_path: t.Optional[str]) -> None:
        if cache_path is None:
            return

        if not os.path.exists(cache_path):
            os.mkdir(cache_path)

        for tensor, name in cache:
            torch.save(tensor, f"{cache_path}/{name}.pt")

    def __len__(self) -> int:
        return len(self.__cache)

    def __getitem__(self, index) -> torch.Tensor:
        return self.__cache[index]
