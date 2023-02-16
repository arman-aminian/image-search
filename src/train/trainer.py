from transformers import Trainer
import torch


def compute_loss(model, inputs, return_outputs=False):
    outputs = model(**inputs, return_loss=True)
    return outputs["loss"]


class CLIPTrainer(Trainer):

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = compute_loss(model, inputs)
        return loss, None, None
