import yaml
from src.base_model.train.model import Corrnet, correlation
import torch.optim as optim
from src.base_model.train.dataset import make_dataset
from src.base_model.train.utils import evaluate_model, print_metrics
import numpy as np
from torch import nn
import torch


def train_base_model(params):
    corrnet = Corrnet(512, 50)
    optimizer = optim.Adam(corrnet.parameters(), lr=0.001)
    model_save_path = params['model_save_path']
    criterion = nn.MSELoss()
    epochs = params['epochs']
    best_val_loss = params['best_val_loss']
    train_dataset, val_dataset, img_test, txt_test = make_dataset(params)

    for e in range(epochs):
        L = []
        err = [[], [], [], []]
        for img, txt in train_dataset:
            # img-> 224*224*3 array
            # txt -> string
            concat_inputs = torch.cat((img, txt), 1)
            optimizer.zero_grad()

            res_combined_input = corrnet(img, txt)
            res_img_input = corrnet(img, torch.zeros_like(txt))
            res_txt_input = corrnet(torch.zeros_like(img), txt)

            err1 = criterion(res_combined_input, concat_inputs)
            err2 = criterion(res_img_input, concat_inputs)
            err3 = criterion(res_txt_input, concat_inputs)
            err4 = correlation(
                corrnet.encoder(img, torch.zeros_like(txt)),
                corrnet.encoder(torch.zeros_like(img), txt)
            )

            loss = (err1 + err2 + err3 + err4)
            loss.backward()
            L.append(loss.item())
            err[0].append(err1.item())
            err[1].append(err2.item())
            err[2].append(err3.item())
            err[3].append(err4.item())
            optimizer.step()

        val_loss = evaluate_model(corrnet, val_dataset, criterion, e)
        print("Epoch: {}:, Train Loss: {}".format(e, np.mean(L)))
        for i in range(len(err)):
            print("err{}: {}".format(i + 1, np.mean(err[i])), end="\t")
        print("\n")

        if e % 5 == 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            print('model_saved')
            torch.save(corrnet.state_dict(), model_save_path)

    print_metrics(corrnet, img_test, txt_test)


if __name__ == '__main__':
    with open("params.yaml", "r") as stream:
        params = yaml.safe_load(stream)

    train_params = params['train']
    train_base_model(train_params)

