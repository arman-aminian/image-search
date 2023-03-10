import torch
from PIL import Image
from src.base_model.train.img2vec import Img2Vec
from src.base_model.train.model import Corrnet
import numpy as np
import pickle
from src.base_model.train.utils import predict
from src.base_model.train.word2vec.preprocessing import Preprocessor
from src.base_model.train.word2vec.utils import text_standardization
from IPython.display import display
from IPython.display import Image as Img


class ImageSearchDemo:
    def __init__(self):
        self.img2vec = Img2Vec('resnet-18', 'default', 512, cuda=True)
        self.w2v_weights = np.load("../train/result/w2v_embedding.npz")['arr_0']
        self.w2v_vocabs = pickle.load(open("../train/result/vocabs.pkl", "rb"))
        self.preprocessor = Preprocessor()
        self.model_save_path = '../train/result/model_state.pt'
        self.corrnet = Corrnet(512, 50)
        self.corrnet.load_state_dict(torch.load(self.model_save_path))
        self.corrnet.eval()

    def compute_text_embedding(self, query: str, embedding_dim=512):
        query_embedding = None
        tf_cleaned_input = text_standardization(query, self.preprocessor)

        v = np.array([0. for i in range(embedding_dim)])
        l = 0
        for word in (tf_cleaned_input.numpy()).decode('utf-8').split():
            word = '[UNK]' if word not in self.w2v_vocabs.keys() else word
            v += self.w2v_weights[self.w2v_vocabs[word]]
            l += 1
        query_embedding = v / l

        return query_embedding

    def image_search(self, query: str, image_name_set, top_num):
        query_embedding = self.compute_text_embedding(query=query)
        img_array = np.zeros((len(image_name_set), 512))
        for i in range(len(image_name_set)):
            img = Image.open(image_name_set[i]).convert('RGB')
            img_array[i] = self.img2vec.get_vec(img)

        txt_array = np.zeros((len(image_name_set), 512))
        for j in range(len(image_name_set)):
            txt_array[j] = query_embedding

        predictions = list(
            predict(self.corrnet, torch.from_numpy(txt_array.astype(np.float32)), torch.from_numpy(img_array.astype(np.float32)))[1])

        predictions_dict = dict(zip(image_name_set, predictions))
        predictions_dict = dict(sorted(predictions_dict.items(), key=lambda item: item[1]))
        if len(predictions_dict.keys()) > top_num:
            return predictions_dict.keys()[0:top_num]
        return predictions_dict.keys()


if __name__ == '__main__':
    image_demo = ImageSearchDemo()
    text = 'خانه ۵۰ متری در شمال تهران'
    image_name_set = [
                        '../data/divar_data/apartment-sell/images/gYH640hU_1.jpg',
                        '../data/divar_data/apartment-sell/images/wY6emgFT_5.jpg',
                        '../data/divar_data/electronic-devices/images/AY1-et6B_1.jpg',
                        '../data/divar_data/entertainment/images/AYtFiWML_2.jpg',
                        '../data/divar_data/home-kitchen/images/AY0lQudu_3.jpg',
                        '../data/divar_data/tools-materials-equipment/images/AY0V6HGG_2.jpg',
                        '../data/divar_data/vehicles/images/AY0CLsXS_1.jpg'
                      ]
    top_num = 3
    predictions_image_pathes = image_demo.image_search(text, image_name_set, top_num)
    for img_path in predictions_image_pathes:
        display(Img(img_path, width=100, height=100))
