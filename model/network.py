
import torch
import logging
import torchvision
from torch import nn

from Team.model.layers import Flatten, L2Norm, GeM

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
    if backbone_name.startswith("resnet"):
        if backbone_name == "resnet18":
            if pretrain in [x.split("_")[-1] for x in PRETRAINED_MODELS.keys()]:
                backbone = get_pretrained_resnet18(backbone_name, pretrain)
            else:
                backbone = torchvision.models.resnet18(pretrained=True)
        elif backbone_name == "resnet50":
            backbone = torchvision.models.resnet50(pretrained=True)
        elif backbone_name == "resnet101":
            backbone = torchvision.models.resnet101(pretrained=True)
        elif backbone_name == "resnet152":
            backbone = torchvision.models.resnet152(pretrained=True)
        
        for name, child in backbone.named_children():
            if name == "layer3":  # Freeze layers before conv_3
                break
            for params in child.parameters():
                params.requires_grad = False
        logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
        layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer
    
    elif backbone_name == "vgg16":
        backbone = torchvision.models.vgg16(pretrained=True)
        layers = list(backbone.features.children())[:-2]  # Remove avg pooling and FC layer
        for layer in layers[:-5]:
            for p in layer.parameters():
                p.requires_grad = False
        logging.debug("Train last layers of the VGG-16, freeze the previous ones")
    
    else:
        raise ValueError(f"Backbone {backbone_name} is not supported")

    backbone = torch.nn.Sequential(*layers)
    
    features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
    
    return backbone, features_dim

def get_pretrained_resnet18(backbone_name, pretrain):
    import os
    import GoogleDriveDownloader as gdd

    if pretrain == 'places':  num_classes = 365
    elif pretrain == 'gldv2':  num_classes = 512
    
    if backbone_name.startswith("resnet18"):
        model = torchvision.models.resnet18(num_classes=num_classes)
    else:
        raise ValueError("This method supports only resnet18, backbone is: {}".format(backbone_name))
    
    model_name = backbone_name.split('conv')[0] + "_" + pretrain
    file_path =  f"content/pretrained/{model_name}.pth"
    

    if not os.path.exists(file_path):
        gdd.download_file_from_google_drive(file_id=PRETRAINED_MODELS[model_name],
                                            dest_path=file_path)
    state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model