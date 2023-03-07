import json
import pickle5 as pickle
import numpy as np
import pandas as pd
import pickle5 as pickle
import yaml
from torch.utils.data import Dataset, DataLoader
import torch
import glob
import pandas as pd
from src.base_model.train.img2vec import Img2Vec
from PIL import Image


class CorrnetDataset(Dataset):

    def __init__(self, img, txt):
        self.img = img
        self.txt = txt
        if self.img.shape[0] != self.txt.shape[0]:
            raise Exception("Different no. of samples")

    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, index):
        _img = self.img[index]
        _txt = self.txt[index]

        return _img, _txt


def read_dataset(file_path):
    dataset_images = pd.DataFrame()
    vector_images = []
    images_name = []
    img2vec = Img2Vec('resnet-18', 'default', 512, cuda=True)

    i = 0
    for image_name in glob.glob(file_path + "/images/*.jpg"):
        i += 1
        img = Image.open(image_name).convert('RGB')
        vector_images.append(img2vec.get_vec(img))
        images_name.append('images/' + image_name.split('\\')[-1])

        if i % 100 == 1:
            print(i)

    dataset_images['image'] = images_name
    dataset_images['img_vec'] = vector_images

    return dataset_images


def make_feature_set(train_param):
    filepath = train_param['dataset_path']

    # dataset = pd.DataFrame()
    # dataset = read_dataset(filepath)
    # dataset.to_pickle(filepath + '/dataset_images')

    # TODO (call function that makes text_datas.csv)
    with open(filepath + '/train_datas', "rb") as fh:
        text_embeddings = pickle.load(fh)

    image_embeddings = pd.read_pickle(filepath + '/dataset_images')
    image_embeddings = image_embeddings.rename({'images_name': 'image'}, axis=1)

    dataset = pd.merge(image_embeddings, text_embeddings, how="inner", on='image')
    dataset = dataset.rename({'vec': 'text_vec'}, axis=1)
    dataset = dataset[['img_vec', 'text_vec']]
    dataset.to_pickle(filepath + '/dataset_img_text_train')


def make_dataset(train_param):
    # make_feature_set(train_param)
    dataset = pd.read_pickle(train_param['dataset_path'] + '/dataset_img_text_train')
    train_image_vectors = np.array(list(dataset['img_vec']))
    train_text_vectors = np.array(list(dataset['text_vec']))

    # dataset = pd.read_pickle(train_param['dataset_path'] + '/dataset_img_text_test')
    # test_image_vectors = np.array(list(dataset['img_vec']))
    # test_text_vectors = np.array(list(dataset['text_vec']))

    val_size = int((train_param['val_size'] / train_param['train_size']) * len(train_image_vectors))
    train_size = len(train_image_vectors) - val_size
    # test_size = len(test_text_vectors)

    img_train = torch.from_numpy(train_image_vectors[:train_size].astype(np.float32))
    txt_train = torch.from_numpy(train_text_vectors[:train_size].astype(np.float32))
    train_dataset = DataLoader(CorrnetDataset(img_train, txt_train), batch_size=train_param['batch_size'], shuffle=True)

    img_val = torch.from_numpy(train_image_vectors[train_size:(train_size + val_size)].astype(np.float32))
    txt_val = torch.from_numpy(train_text_vectors[train_size:(train_size + val_size)].astype(np.float32))
    val_dataset = DataLoader(CorrnetDataset(img_val, txt_val), batch_size=train_param['batch_size'], shuffle=True)

    # img_test = torch.from_numpy(image_vectors[(train_size + val_size):].astype(np.float32))
    # txt_test = torch.from_numpy(text_vectors[(train_size + val_size):].astype(np.float32))
    # test_dataset = DataLoader(CorrnetDataset(img_test, txt_test), batch_size=train_param['batch_size'], shuffle=True)
    img_test = None
    txt_test = None

    return train_dataset, val_dataset, img_test, txt_test
