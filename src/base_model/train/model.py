import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, input_vec_dim):
        super(Encoder, self).__init__()

        self.FC1_img = nn.Linear(input_vec_dim, 300)
        self.FC2_img = nn.Linear(300, 200)
        self.FC3_img = nn.Linear(200, 50)

        self.FC1_txt = nn.Linear(input_vec_dim, 300)
        self.FC2_txt = nn.Linear(300, 200)
        self.FC3_txt = nn.Linear(200, 50)

    def forward(self, img, txt):
        x = F.relu(self.FC1_img(img))
        x = F.relu(self.FC2_img(x))
        x = F.relu(self.FC3_img(x))

        y = F.relu(self.FC1_txt(txt))
        y = F.relu(self.FC2_txt(y))
        y = F.relu(self.FC3_txt(y))

        return F.relu(torch.add(x, y))


class Decoder(nn.Module):

    def __init__(self, output_vec_dim):
        super(Decoder, self).__init__()

        self.FC1_img = nn.Linear(output_vec_dim, 200)
        self.FC2_img = nn.Linear(200, 300)
        self.FC3_img = nn.Linear(300, 512)

        self.FC1_txt = nn.Linear(output_vec_dim, 200)
        self.FC2_txt = nn.Linear(200, 300)
        self.FC3_txt = nn.Linear(300, 512)

    def forward(self, rep):
        x = F.relu(self.FC1_img(rep))
        x = F.relu(self.FC2_img(x))
        x = F.relu(self.FC3_img(x))

        y = F.relu(self.FC1_txt(rep))
        y = F.relu(self.FC2_txt(y))
        y = F.relu(self.FC3_txt(y))

        combined = F.relu(torch.cat((x, y), 1))
        return combined


class Corrnet(nn.Module):
    def __init__(self, input_vec_dim, latent_rep_dim):
        super(Corrnet, self).__init__()
        self.encoder = Encoder(input_vec_dim)
        self.decoder = Decoder(latent_rep_dim)

    def forward(self, img, txt):
        latent_rep = self.encoder(img, txt)
        combined = self.decoder(latent_rep)
        return combined


def correlation(x, y, lamda=0.02):
    '''
      x, y are n x 50 dimensional vectors obtained from the respective n x 512 embeddings
    '''

    x_mean = torch.mean(x, dim=0)  # Along the y-axis, that is, average of all feature vectors
    y_mean = torch.mean(y, dim=0)  # 1 x 50 dimensional
    x_centered = torch.sub(x, x_mean)  # calculates xi - X_mean n x 50 dimensional
    y_centered = torch.sub(y, y_mean)  # calculates yi - Y_mean
    corr_nr = torch.sum(torch.mul(x_centered, y_centered))  # The numerator
    # print(list(corr_nr.shape))
    corr_dr1 = torch.sqrt(torch.sum(torch.square(x_centered)))
    corr_dr2 = torch.sqrt(torch.sum(torch.square(y_centered)))
    corr_dr = corr_dr1 * corr_dr2
    corr = -lamda * corr_nr / corr_dr
    # print(corr.item()) # Should decrease ideally
    return corr
