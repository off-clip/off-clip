import torch
import torch.nn as nn
import torchvision.transforms as transforms
from . import models
from . import loss
from offclip.constants import *

def build_offclip_model(cfg):
    CARZero_model = models.offclip_model.OFFCLIP(cfg)
    return CARZero_model

def build_dqn_wo_self_atten_mlp_module(cfg):
    fusion = models.dqn_wo_self_atten_mlp.TQN_Model(cfg)
    return fusion

def build_img_model(cfg):
    image_model = models.IMAGE_MODELS[cfg.phase.lower()]
    return image_model(cfg)

def build_text_model(cfg):
    return models.text_model.BertEncoder(cfg)


def build_fusion_module(cfg):
    fusion = models.fusion_module.Fusion(cfg)
    return fusion

def build_dqn_module(cfg):
    fusion = models.dqn.TQN_Model(cfg)
    return fusion

def build_gpt_model(cfg):
    return models.gpt_model.EmbeddingFusing(cfg)



def build_loss(cfg):

    if cfg.train.loss_fn.type == "DiceLoss":
        return loss.segmentation_loss.DiceLoss()
    elif cfg.train.loss_fn.type == "FocalLoss":
        return loss.segmentation_loss.FocalLoss()
    elif cfg.train.loss_fn.type == "MixedLoss":
        return loss.segmentation_loss.MixedLoss(alpha=cfg.train.loss_fn.alpha)
    elif cfg.train.loss_fn.type == "BCE":
        if cfg.train.loss_fn.class_weights is not None:
            weight = torch.Tensor(cfg.train.loss_fn.class_weights)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            loss_fn = nn.BCEWithLogitsLoss()
        return loss_fn
    else:
        raise NotImplementedError(f"{cfg.train.loss_fn} not implemented yet")


def build_transformation(cfg, split):

    t = []
    if split == "train":

        if cfg.transforms.random_crop is not None:
            t.append(transforms.RandomCrop(cfg.transforms.random_crop.crop_size))

        if cfg.transforms.random_horizontal_flip is not None:
            t.append(
                transforms.RandomHorizontalFlip(p=cfg.transforms.random_horizontal_flip)
            )

        if cfg.transforms.random_affine is not None:
            t.append(
                transforms.RandomAffine(
                    cfg.transforms.random_affine.degrees,
                    translate=[*cfg.transforms.random_affine.translate],
                    scale=[*cfg.transforms.random_affine.scale],
                )
            )

        if cfg.transforms.color_jitter is not None:
            t.append(
                transforms.ColorJitter(
                    brightness=[*cfg.transforms.color_jitter.bightness],
                    contrast=[*cfg.transforms.color_jitter.contrast],
                )
            )
    else:
        if cfg.transforms.random_crop is not None:
            t.append(transforms.CenterCrop(cfg.transforms.random_crop.crop_size))

    t.append(transforms.ToTensor())
    if cfg.transforms.norm is not None:
        if cfg.transforms.norm == "imagenet":
            t.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        elif cfg.transforms.norm == "half":
            t.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        elif cfg.transforms.norm == "CXR_MAE":
            t.append(transforms.Normalize(mean=[0.4978], std=[0.2449]))
        elif cfg.transforms.norm == "CLIP":
            t.append(transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)))
        else:
            raise NotImplementedError("Normaliation method not implemented")

    return transforms.Compose(t)
