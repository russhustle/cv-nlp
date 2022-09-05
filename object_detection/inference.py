from transformers import DetrFeatureExtractor
from transformers import DetrForObjectDetection
from omegaconf import DictConfig
import torch


def inference(img, cfg: DictConfig):
    feature_extractor = DetrFeatureExtractor.from_pretrained(
        cfg.feature_extractor
    )
    encoding = feature_extractor(img, return_tensors="pt")
    model = DetrForObjectDetection.from_pretrained(cfg.model)
    outputs = model(**encoding)
    return feature_extractor, model, encoding, outputs


def rescale_bbox(img, outputs, feature_extractor, cfg: DictConfig):
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > cfg.keep_threshold
    target_sizes = torch.tensor(img.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(
        outputs, target_sizes
    )
    bboxes_scaled = postprocessed_outputs[0]["boxes"][keep]
    return probas, keep, bboxes_scaled


def get_attention_weights(model, encoding):
    # use lists to store the outputs via up-values
    conv_features = []
    hooks = [
        model.model.backbone.conv_encoder.register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
    ]
    # propagate through the model
    outputs = model(**encoding, output_attentions=True)
    for hook in hooks:
        hook.remove()
    # don't need the list anymore
    conv_features = conv_features[0]
    # get cross-attention weights of last decoder layer - which is of shape (batch_size, num_heads, num_queries, width*height)
    dec_attn_weights = outputs.cross_attentions[-1]
    # average them over the 8 heads and detach from graph
    dec_attn_weights = torch.mean(dec_attn_weights, dim=1).detach()
    return conv_features, dec_attn_weights
