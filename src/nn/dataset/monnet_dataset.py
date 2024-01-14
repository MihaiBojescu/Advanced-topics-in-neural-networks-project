import os
import torch
import typing as t
from torch.utils.data import Dataset
from torchvision.transforms.v2 import ToTensor
from PIL import Image


class MonnetDataset(Dataset):
    __transforms: t.Optional[t.Callable[[torch.Tensor], torch.Tensor]]
    __data: t.List[torch.Tensor]

    def __init__(
        self,
        *args,
        data_path: str,
        transforms: t.Optional[t.Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(*args)

        self.__transforms = transforms
        self.__data = self.__load_data(data_path)

    def __load_data(self, data_path: str) -> t.List[torch.Tensor]:
        data = []
        transform = ToTensor()
        files = os.listdir(data_path)

        for file in files:
            image = Image.open(f"{data_path}/{file}")
            tensor = transform(image)
            data.append(tensor)

        return data

    def __len__(self) -> int:
        return len(self.__data)

    def __getitem__(self, index) -> torch.Tensor:
        data = self.__data[index]

        if self.__transforms:
            data = self.__transforms(data)

        return data