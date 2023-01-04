
import torch
import logging
import torchvision
from torch import nn

from .layers import Flatten, L2Norm, GeM
from .cct import cct_14_7x2_384

# Pretrained models on Google Landmarks v2 and Places 365
PRETRAINED_MODELS = {
        'resnet18_places'  : '1DnEQXhmPxtBUrRc81nAvT8z17bk-GBj5',
        'resnet18_gldv2'   : '1wkUeUXFXuPHuEvGTXVpuP5BMB-JJ1xke',
    }

CHANNELS_NUM_IN_LAST_CONV = {
        "resnet18": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "resnet152": 2048,
        "vgg16": 512,
        "efficientnet_b0": 1280,
        "efficientnet_b1": 1280,
        "efficientnet_b2": 1408,
        "efficientnet_v2_s": 1280,
        "mobilenet_v3_small": 576,
        "mobilenet_v3_large": 960,
        "convnext_tiny": 768,
        "swin_tiny": 768,
        "cct384": 384,
    }


class GeoLocalizationNet(nn.Module):
    def __init__(self, backbone, fc_output_dim, pretrain):
        super().__init__()
        self.backbone, features_dim = get_backbone(backbone, pretrain)
        self.aggregation = nn.Sequential(
                L2Norm(),
                GeM(),
                Flatten(),
                nn.Linear(features_dim, fc_output_dim),
                L2Norm()
            )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x

def get_backbone(backbone_name, pretrain):
    weights='IMAGENET1K_V1'
    if backbone_name.startswith("resnet"):
        if backbone_name == "resnet18":
            logging.info(f"Resnet18 pretrained on: {pretrain}")
            if pretrain in [x.split("_")[-1] for x in PRETRAINED_MODELS.keys()]:
                backbone = get_pretrained_resnet18(backbone_name, pretrain)
            else:
                backbone = torchvision.models.resnet18(weights=weights)
        elif backbone_name == "resnet50":
            backbone = torchvision.models.resnet50(weights=weights)
        elif backbone_name == "resnet101":
            backbone = torchvision.models.resnet101(weights=weights)
        elif backbone_name == "resnet152":
            backbone = torchvision.models.resnet152(weights=weights)
        
        for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer
    
    elif backbone_name == "vgg16":
        backbone = torchvision.models.vgg16(weights=weights)
        layers = list(backbone.features.children())[:-2]  # Remove avg pooling and FC layer
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last layers of the VGG-16, freeze the previous ones")
    
    elif backbone_name.startswith("efficientnet"):
        if backbone_name == "efficientnet_b0":
            backbone = torchvision.models.efficientnet_b0(weights=weights)
        elif backbone_name == "efficientnet_b1":
            backbone = torchvision.models.efficientnet_b1(weights=weights)
        elif backbone_name == "efficientnet_b2":
            backbone = torchvision.models.efficientnet_b2(weights=weights)
        elif backbone_name == "efficientnet_v2_s":
            backbone = torchvision.models.efficientnet_v2_s(weights=weights)
        
        layers = list(backbone.features.children()) # Remove avg pooling and FC layer
        for layer in layers[:-2]: # freeze all the layers except the last two
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last two layers of EfficientNet, freeze the previous ones")
    
    elif backbone_name.startswith("mobilenet"):
        if backbone_name == "mobilenet_v3_small":
            backbone = torchvision.models.mobilenet_v3_small(weights=weights)
        elif backbone_name == "mobilenet_v3_large":
            backbone = torchvision.models.mobilenet_v3_large(weights=weights)

        layers = list(backbone.features.children()) # Remove avg pooling and FC layer
        # TODO consider to freeze up to layers[:-3]
        for layer in layers[:-3]: # freeze all the layers except the last two
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last two layers of MobileNet, freeze the previous ones")

    elif backbone_name.startswith("convnext"):
        if backbone_name == "convnext_tiny":
            backbone = torchvision.models.convnext_tiny(weights=weights)
        layers = list(backbone.features.children()) # Remove avg pooling and FC layer
        for layer in layers[:-1]: # freeze all the layers except the last one
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last layer of ConvNext, freeze the previous ones")

    elif backbone_name.startswith("swin"):
        if backbone_name == "swin_tiny":
            backbone = torchvision.models.swin_t(weights=weights)
        # TODO consider to get just the features layers
        layers = list(backbone.children())[:-3] # Remove avg pooling and FC layer
        for x in layers[0][:-1]:
            for p in x.parameters():
                p.requires_grad = False # freeze all the layers except the last three blocks
        logging.debug("Train last three layers of Swin, freeze the previous ones")

    elif backbone.startswith("cct"):
        if backbone.startswith("cct384"):
            backbone = cct_14_7x2_384(pretrained=True, progress=True, aggregation="seqpool")

        trunc_te = 8        # value from 04/01 Q&A 
        freeze_te = 1       # value from 04/01 Q&A

        logging.debug(f"Truncate CCT at transformers encoder {trunc_te}")
        backbone.classifier.blocks = torch.nn.ModuleList(backbone.classifier.blocks[:trunc_te].children())
        logging.debug(f"Freeze all the layers up to tranformer encoder {freeze_te}")
        for p in backbone.parameters():
            p.requires_grad = False
        for name, child in backbone.classifier.blocks.named_children():
            if int(name) > freeze_te:
                for params in child.parameters():
                    params.requires_grad = True

        features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
        return backbone, features_dim

    else:
        raise ValueError(f"Backbone {backbone_name} is not supported")

    backbone = torch.nn.Sequential(*layers)
    
    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
    
    return backbone, features_dim

def get_pretrained_resnet18(backbone_name, pretrain):
    import os
    import gdown
    def download(id, output=None, quiet=True):
        gdown.download(
            f"https://drive.google.com/uc?export=download&confirm=pbef&id={id}",
            output=output,
            quiet=quiet
        )


    if pretrain == 'places':  num_classes = 365
    elif pretrain == 'gldv2':  num_classes = 512
    
    if backbone_name.startswith("resnet18"):
        model = torchvision.models.resnet18(num_classes=num_classes)
    else:
        raise ValueError("This method supports only resnet18, backbone is: {}".format(backbone_name))
    
    model_name = backbone_name.split('conv')[0] + "_" + pretrain
    file_path =  f"/content/pretrained/{model_name}.pth"
    

    if not os.path.exists(file_path):
        download(id=PRETRAINED_MODELS[model_name], output=file_path)
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model