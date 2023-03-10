import sys
from transformers import CLIPModel, CLIPConfig, CLIPVisionModel
from transformers import AutoModel, AutoTokenizer
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
from .acc_at_k import calculate_pairwise_similarity, calculate_accuracy_at_k, plot_accuracy_at_k


def to_device(x, device="cuda:0"):
    if isinstance(x, dict):
        return {k: to_device(v) for k, v in x.items()}
    return x.to(device=device)


def load_clip_models():
    text_tokenizer = AutoTokenizer.from_pretrained('arman-aminian/farsi-image-search-text')
    text_encoder = AutoModel.from_pretrained('arman-aminian/farsi-image-search-text').to(device='cuda:0')
    image_encoder = CLIPVisionModel.from_pretrained('arman-aminian/farsi-image-search-vision').to(device='cuda:0')

    return text_tokenizer, text_encoder, image_encoder


def calc_embedding_for_text(text, text_tokenizer, text_encoder):
    with torch.no_grad():
        tokenized = text_tokenizer(text, return_tensors='pt')
        embedding = text_encoder(**to_device(tokenized)).pooler_output
    return embedding.squeeze().cpu().tolist()


def calc_embedding_for_image(image_path, image_encoder):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]),
        )
    ])
    image = preprocess(Image.open(image_path).convert('RGB'))
    image = image.unsqueeze(0)
    with torch.no_grad():
        embedding = image_encoder(to_device(image)).pooler_output
    return embedding.squeeze().cpu().tolist()


def calculate_embeddings(df):
    text_embeddings = []
    for i in tqdm(range(1000)):
        text_embeddings.append(calc_embedding_for_text(df.iloc[i]['translation']))

    image_embeddings = []
    for i in tqdm(range(1000)):
        image_embeddings.append(
            calc_embedding_for_image('/content/flickr30k_images/flickr30k_images/' + df.iloc[i]['image_name']))

    return text_embeddings, image_embeddings


def main(flickr_df_path):
    df = pd.read_csv(flickr_df_path)
    text_embeddings, image_embeddings = calculate_embeddings(df)
    cosine_matrix = calculate_pairwise_similarity(text_embeddings, image_embeddings)
    accuracy_at = calculate_accuracy_at_k(cosine_matrix)
    plot_accuracy_at_k(accuracy_at)


if __name__ == '__main__':
    main(sys.argv[0])
