from transformers import CLIPModel, CLIPConfig, CLIPVisionModel
from transformers import AutoModel


def get_clip_model(image_embedding_model, text_embedding_model, config):
    clip_config = CLIPConfig(text_config_dict=config,
                             vision_config_dict=config)
    clip = CLIPModel(config=clip_config)
    clip.text_projection = AutoModel.from_pretrained(text_embedding_model)
    clip.visual_projection = CLIPVisionModel.from_pretrained(image_embedding_model)
    return clip
