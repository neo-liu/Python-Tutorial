import os
from PIL import Image
from torch.utils.data import Dataset


class Train_Data(Dataset):
    def __init__(self, path1, path2, transform=None):
        if not os.path.isdir(path1):
            raise ValueError("input image_path is not a dir")
        elif not os.path.isdir(path2):
            raise ValueError("input label_path is not a dir")

        self.image_path = path1
        self.image_list = os.listdir(path1)
        self.label_path = path2
        self.label_list = os.listdir(path2)

        self.transform = transform # transforms.ToTensor()

    # def _read_convert_image(self, image_name):
    #     #     image = Image.open(image_name)
    #     #     image = self.transforms(image).float()
    #     #     return image

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # 注意这里读到的图片的通道数！！！
        image_name = os.path.join(self.image_path, self.image_list[index])
        label_name = os.path.join(self.label_path, self.label_list[index])

        image = Image.open(image_name)
        label = Image.open(label_name)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return label, image


class Test_Data(Dataset):
    def __init__(self, path1, path2, transform=None):
        if not os.path.isdir(path1):
            raise ValueError("input image_path is not a dir")
        elif not os.path.isdir(path2):
            raise ValueError("input label_path is not a dir")

        self.image_path = path1
        self.image_list = os.listdir(path1)
        self.label_path = path2
        self.label_list = os.listdir(path2)

        self.transform = transform # transforms.ToTensor()

    # def _read_convert_image(self, image_name):
    #     #     image = Image.open(image_name)
    #     #     image = self.transforms(image).float()
    #     #     return image

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        # 注意这里读到的图片的通道数！！！
        image_name = os.path.join(self.image_path, self.image_list[index])
        label_name = os.path.join(self.label_path, self.label_list[index])

        image = Image.open(image_name)
        label = Image.open(label_name)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return label, image