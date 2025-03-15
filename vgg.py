import os
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG16_Weights


# Taken from https://github.com/explainingai-code/StableDiffusion-PyTorch/blob/main/models/lpips.py
# Where that itself was taken from https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py
# Based on the paper "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


class VGG(torch.nn.Module):
    def __init__(self):
        """
        Extracts features from selected VGG layers.
        """

        super().__init__()
        # Load pretrained vgg model from torchvision
        pretrained_vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

        # Take slices out of the VGG model to evaluate intermediate results
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), pretrained_vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), pretrained_vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), pretrained_vgg[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), pretrained_vgg[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), pretrained_vgg[x])


    def forward(self, x):
        # Return output of vgg features
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h

        # Return the results
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class LPIPS(nn.Module):
    def __init__(self, lpips_weight_path, device):
        """
        Used to return the perceptual loss between original and predicted image

        :param lpips_weight_path: Path to the model weights based on LPIPS
        :param device: Device to train on (cpu or cuda)
        """

        super(LPIPS, self).__init__()
        self.scaling_layer = ScalingLayer()

        # Init channels and VGG model
        self.channels = [64, 128, 256, 512, 512]
        self.net = VGG()

        self.lin0 = NetLinLayer(self.channels[0])
        self.lin1 = NetLinLayer(self.channels[1])
        self.lin2 = NetLinLayer(self.channels[2])
        self.lin3 = NetLinLayer(self.channels[3])
        self.lin4 = NetLinLayer(self.channels[4])
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        self.lins = nn.ModuleList(self.lins)

        if os.path.exists(lpips_weight_path):
            self.load_state_dict(torch.load(lpips_weight_path, map_location=device), strict=False)
        else:
            raise Exception(f"Path {lpips_weight_path=} is not found!")


    def forward(self, clean, pred):
        clean_min, clean_max = clean.min(), clean.max()
        pred_min, pred_max = pred.min(), pred.max()

        # Emm...kind of lengthy and possibly redundant at some areas. Though it's fine lol
        if clean.shape != pred.shape:
            raise ValueError(f"Input tensors 'clean' and 'pred' must have the same shape. {clean.shape=}, {pred.shape=}")

        # Next normalize to range [-1, 1] if in range [0, 1]
        elif clean_min >= 0 and clean_max <= 1:
            if not (pred_min >= 0 and pred_max <= 1):
                raise ValueError(f"Clean tensor is in range [0, 1] but pred tensor is in range [{pred_min}, {pred_max}]")
            clean = (clean * 2) - 1
            pred = (pred * 2) - 1
        elif pred_min >= 0 and pred_max <= 1:
            if not (clean_min >= 0 and clean_max <= 1):
                raise ValueError(f"Pred tensor is in range [0, 1] but clean tensor is in range [{clean_min}, {clean_max}]")
            clean = (clean * 2) - 1
            pred = (pred * 2) - 1

        elif clean_min >= -1 and clean_max <= 1:
            if not (pred_min >= -1 and pred_max <= 1):
                raise ValueError(f"Clean tensor is in range [-1, 1] but pred tensor is in range [{pred_min}, {pred_max}]")
        elif pred_min >= -1 and pred_max <= 1:
            if not (clean_min >= -1 and clean_max <= 1):
                raise ValueError(f"Pred tensor is in range [-1, 1] but clean tensor is in range [{clean_min}, {clean_max}]")

        else:
            raise ValueError(f"Tensors must be normalized to the range [-1, 1]. Instead, {clean_min=}, {clean_max=}, {pred_min=}, {pred_max=}")


        clean_input, pred_input = self.scaling_layer(clean), self.scaling_layer(pred)

        # Get VGG output between clean and predicted tensors
        out_clean, out_pred = self.net(clean_input), self.net(pred_input)
        feats0, feats1, diffs = {}, {}, {}

        # Difference is calculated as difference of squares
        for kk in range(len(self.channels)):
            feats0[kk], feats1[kk] = F.normalize(out_clean[kk]), F.normalize(out_pred[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        # 1x1 convolution followed by spatial average on the square differences
        res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(len(self.channels))]

        return sum(res)


class ScalingLayer(nn.Module):
    def __init__(self):
        """
        Used to normalize/scale given tensors
        """

        super(ScalingLayer, self).__init__()
        # To normalize inputs. Shift corresponds to mean, scale corresponds to variance
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    def __init__(self, in_channel, out_channel=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(in_channel, out_channel, 1, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        return self.model(x)
