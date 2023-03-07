import torch
from src.base_model.train.model import correlation
import numpy as np
from statistics import median
from scipy.spatial import distance


def evaluate_model(model, val_dataset, criterion, e):
    model.eval()
    L = []
    err = [[], [], [], []]
    for img, txt in val_dataset:
        # img-> 224*224*3 array
        # txt -> string
        concat_inputs = torch.cat((img, txt), 1)

        res_combined_input = model(img, txt)
        res_img_input = model(img, torch.zeros_like(txt))
        res_txt_input = model(torch.zeros_like(img), txt)

        err1 = criterion(res_combined_input, concat_inputs)
        err2 = criterion(res_img_input, concat_inputs)
        err3 = criterion(res_txt_input, concat_inputs)
        err4 = correlation(
            model.encoder(img, torch.zeros_like(txt)),
            model.encoder(torch.zeros_like(img), txt)
        )

        loss = (err1 + err2 + err3 + err4)

        L.append(loss.item())
        err[0].append(err1.item())
        err[1].append(err2.item())
        err[2].append(err3.item())
        err[3].append(err4.item())

    print("Epoch: {}:, Val Loss: {}".format(e, np.mean(L)))
    for i in range(len(err)):
        print("err{}: {}".format(i + 1, np.mean(err[i])), end="\t")
    print('/n')
    model.train()
    return np.mean(L)


def predict(corrnet, img, txt):
    img_vecs = corrnet.encoder(img, torch.zeros_like(txt))
    txt_vecs = corrnet.encoder(torch.zeros_like(img), txt)

    euc = []
    cos = []
    for img_vec, txt_vec in zip(img_vecs, txt_vecs):
        euc.append(distance.euclidean(img_vec.cpu().detach().numpy(), txt_vec.cpu().detach().numpy()))
        cos.append(distance.cosine(img_vec.cpu().detach().numpy(), txt_vec.cpu().detach().numpy()))

    return np.array(euc), np.array(cos)


def print_metrics(corrnet, img_test, txt_test):
    mr = []
    top_1_count = 0
    top_5_count = 0
    top_10_count = 0
    test_size = len(img_test)
    for i in range(test_size):
        img_array = np.zeros((test_size, 512))
        for k in range(test_size):
            img_array[k] = img_test[k]

        txt_array = np.zeros((test_size, 512))
        for j in range(test_size):
            txt_array[j] = txt_test[i]

        predictions = list(
            predict(corrnet, torch.from_numpy(txt_array.astype(np.float32)), torch.from_numpy(img_array.astype(np.float32)))[1])
        pred_i = predictions[i]
        predictions.sort()
        rank = predictions.index(pred_i)
        if rank < 10:
            top_10_count += 1
        if rank < 5:
            top_5_count += 1
        if rank < 1:
            top_1_count += 1
        mr.append(rank + 1)

    print('Median Rank(txt->img):', median(mr) * 100 / test_size, '%')
    print('R@1(txt->img):', top_1_count * 100 / test_size, '%')
    print('R@5(txt->img):', top_5_count * 100 / test_size, '%')
    print('R@10(txt->img):', top_10_count * 100 / test_size, '%')
