import sys
import os
from shutil import copyfile

sys.path.append("../..")  # Adds higher directory to python modules path.

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image

import numpy as np

import pickle

architectures = {'resnet-18': 512, 'alexnet': 4096,
                 'resnet-152': 2048, 'vgg19': 4096}

""" *********************************
Img2Vec class
*********************************** """


class Img2Vec():

    def __init__(self, model='vgg19', layer='default',
                 layer_output_size=4096, cuda=False):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        self.cuda = cuda
        self.layer_output_size = layer_output_size
        self.model_name = model
        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)

        if self.cuda:
            self.model.cuda()

        self.model.eval()

        self.scaler = transforms.Resize((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def _get_model_and_layer(self, model_name, layer):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """
        if model_name == 'resnet-18':
            model = models.resnet18(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'resnet-152':
            model = models.resnet152(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 2048
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        elif model_name == 'vgg19':
            model = models.vgg19(pretrained=True)
            if layer == 'default':
                layer = model.classifier[-3]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        else:
            raise KeyError('Model %s was not found' % model_name)

    """ *********************************************************

    Img2Vec methods

    ********************************************************* """

    def get_vec(self, img, tensor=False):
        """ Get vector embedding from PIL image
        :param img: PIL Image
        :param tensor: If True, get_vec will return a FloatTensor
                       instead of Numpy array
        :returns: Numpy ndarray
        """
        if self.cuda:
            image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0)
            image = Variable(image).cuda()
        else:
            image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0)
            image = Variable(image)

        my_embedding = torch.zeros(self.layer_output_size)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data.reshape(self.layer_output_size))

        h = self.extraction_layer.register_forward_hook(copy_data)
        h_x = self.model(image)
        h.remove()

        if tensor:
            return my_embedding
        else:
            return my_embedding.numpy()

    def get_matrix_from_folder(self, input_path, sample_size=None):
        """ Get a matrix with the vector embeddings of all
            the images contained in a folder.
        input_path : Input folder
        sample_size : Uses just a random sample fromm all files
        returns: numpy array with embeddings, list of processed files
        """
        files = [f for f in os.listdir(input_path) \
                 if os.path.isfile(os.path.join(input_path, f))]

        if sample_size:
            sample_indices = np.random.choice(range(0, len(files)),
                                              size=sample_size,
                                              replace=False)
        else:
            sample_size = len(files)
            sample_indices = list(range(sample_size))

        vec_mat = np.zeros((sample_size, self.layer_output_size))

        for index, i in enumerate(sample_indices):
            file = files[i]
            filename = os.fsdecode(file)
            img = Image.open(os.path.join(input_path, filename))
            vec = self.get_vec(img)
            vec_mat[index, :] = vec

        return vec_mat, files

    def pickle_matrix_from_folder(self, input_path, output_path=None,
                                  sample_size=None):
        """ Pickles a matrix with the vector embeddings of all
            the images contained in a folder.
        input_path : Input folder
        output_path : Output folder for pickles (default: same as input)
        sample_size : Uses just a random sample fromm all files
        """
        output_path = output_path or input_path

        files = [f for f in os.listdir(input_path) \
                 if os.path.isfile(os.path.join(input_path, f))]

        if sample_size:
            sample_indices = np.random.choice(range(0, len(files)),
                                              size=sample_size,
                                              replace=False)
        else:
            sample_size = len(files)
            sample_indices = list(range(sample_size))

        vec_mat = np.zeros((sample_size, self.layer_output_size))

        for index, i in enumerate(sample_indices):
            file = files[i]
            filename = os.fsdecode(file)
            img = Image.open(os.path.join(input_path, filename))
            vec = self.get_vec(img)
            vec_mat[index, :] = vec

        output_name = output_path + '/' + self.model_name + '_img_names.pkl'
        print('Processed filenames: ' + output_name)
        pickle.dump(files, open(output_name, 'wb'))

        output_name = output_path + '/' + self.model_name + '_img_vecs.pkl'
        print('Embedded vectors: ' + output_name)
        pickle.dump(vec_mat, open(output_name, 'wb'))


""" *********************************
Useful functions
*********************************** """


def img2vec_from_folder_to_pickle(input_path, output_path=None,
                                  model='vgg19', layer='default',
                                  sample_size=None):
    """ Pickles a matrix with the vector embeddings of all
        the images contained in a folder.
    input_path : Input folder
    model : NN architecture
    layer : Output layer
    output_path : Output folder for pickles (default: same as input)
    sample_size : Uses just a random sample fromm all files
    """

    if model in architectures:
        img2vec = Img2Vec(model=model, layer=layer)
    else:
        raise ("Invalid NN architecture")

    img2vec.pickle_matrix_from_folder(input_path, output_path=output_path,
                                      sample_size=sample_size)


def img2vec_from_folder_to_matrix(input_path, output_path=None,
                                  model='vgg19', layer='default',
                                  sample_size=None):
    """ Pickles a matrix with the vector embeddings of all
        the images contained in a folder.
    input_path : Input folder
    model : NN architecture
    layer : Output layer
    output_path : Output folder for pickles (default: same as input)
    sample_size : Uses just a random sample fromm all files
    returns: numpy array with embeddings, list of processed files
    """

    if model in architectures:
        img2vec = Img2Vec(model=model, layer=layer)
    else:
        raise KeyError('Model %s was not found' % model_name)

    return img2vec.get_matrix_from_folder(input_path,
                                          sample_size=sample_size)

