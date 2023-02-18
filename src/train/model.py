from transformers import CLIPModel, CLIPConfig


def get_clip_model(image_embedding_model, text_embedding_model, config):
    clip_config = CLIPConfig(text_config_dict=config,
                             vision_config_dict=config)
    clip = CLIPModel(config=clip_config)
    clip.text_projection = text_embedding_model
    clip.visual_projection = image_embedding_model
    return clip
