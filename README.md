<span align="center">
<a href="https://huggingface.co/spaces/arman-aminian/farsi-image-search"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Demo&color=orange"></a>
<a href="https://huggingface.co/arman-aminian/farsi-image-search-text"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Text Models&color=blue"></a>
<a href="https://huggingface.co/arman-aminian/farsi-image-search-vision"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Vision Models&color=blue"></a>
<a href="https://colab.research.google.com/drive/1QZWgNFy9NZxj2g4pQH_UDFVVg2eGUOrn?usp=sharing"><img src="https://img.shields.io/static/v1?label=Colab&message=Training notebook&logo=Google%20Colab&color=purple"></a>
<a href="https://colab.research.google.com/drive/1bsKbW0URSJGWwS6Nu33X_l3tHsLpqsF7?usp=sharing"><img src="https://img.shields.io/static/v1?label=Colab&message=Inference notebook&logo=Google%20Colab&color=red"></a>
</span>

# image-search

## CLIP

[CLIP](https://arxiv.org/abs/2103.00020) (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3.

With CLIP, we can train any two image and text encoder models together to relate images and text. It gives a score for relatedness of any given text and image! We fine-tuned [Vision Transformer(ViT)](https://huggingface.co/openai/clip-vit-base-patch32) as the vision encoder and the [roberta-zwnj-wnli-mean-tokens](https://huggingface.co/m3hrdadfi/roberta-zwnj-wnli-mean-tokens) as the farsi text encoder.

## Translation

There weren't datasets with Persian captioned images, so we translated datasets with English captions to Persian with Google Translate using [googletrans](https://pypi.org/project/googletrans/) python package.

Then we evaluated these translations with a [sentence-bert](https://www.sbert.net/) bilingual model named [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2) trained for sentence similarity.
We calculated cosine similarity for embeddings of English caption and its Persian translation. The histogram of this score is shown below:

<img alt="translation-score.png" src="images/translation-score.png" width="400"/>

Finally, we filtered out top translations. Some samples of the final dataframe:

<img alt="translation-sample-df.png" src="images/translation-sample-df.png" width="700"/>

More details of translation part can be found in [this notebook](https://colab.research.google.com/drive/1XcwbdegPsuXKybDczD4d8d8LLH1QlQ8m).



## Evaluation

### Accuracy @ k

This metric is used for evaluating how good an image search of a model is.

Acc@k definition: Is the best image (the most related to the text query), among the top-k outputs of the model?

We calculated this metric for both models (CLIP & baseline) on two datasets:
* [flickr30k](https://paperswithcode.com/dataset/flickr30k): some intersections with the training data.
* [nocaps](https://nocaps.org/): completely zero-shot for models!

We can see the results of our CLIP model on a sample of flickr dataset with size 1000 (the right diagram has a log scale in its x-axis:

<img alt="clip-flickr" src="./images/clip-flickr.png" width="35%"/> <img alt="clip-flickr-log" src="./images/clip-flickr-log.png" width="35%"/>

And hear are the results of our CLIP model on a sample of nocaps dataset with size 1000 (the right diagram has a log scale in its x-axis:

<img alt="clip-nocaps" src="./images/clip-nocaps.png" width="35%"/> <img alt="clip-nocaps-log" src="./images/clip-nocaps-log.png" width="35%"/>

You can find more details in notebooks for [CLIP evaluation](https://colab.research.google.com/drive/1Rj9gFo4pTo1p-H2G3uw1viTJVJ8_-ZUF) and [baseline evaluation](https://colab.research.google.com/drive/13NwD0bE0JaR5L6fj26EyoDVhB5G7lgfX)

