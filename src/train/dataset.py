import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class CLIPDataset(Dataset):
    def __init__(self,
                 image_paths: list,
                 text: list,
                 tokenizer,
                 max_len,
                 image_size,
                 image_mean,
                 image_std,
                 mode: str = 'train'):
        self.image_paths = image_paths
        self.tokens = tokenizer(text, padding='max_length',
                                max_length=max_len,
                                truncation=True)

        if mode == 'train':
            self.augment = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=image_mean, std=image_std)
            ])
        elif mode == 'test':
            self.augment = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=image_mean, std=image_std)
            ])
