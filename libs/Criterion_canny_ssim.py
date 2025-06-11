import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import cv2
import numpy as np
from libs.ssim import get_default_ssim_image_filter, calculate_ssim_repr
from libs.ssim import calculate_simplified_ssim_ctx, calculate_score

class styleLoss(nn.Module):
    def forward(self, input, target):
        ib, ic, ih, iw = input.size()
        iF = input.view(ib, ic, -1)
        iMean = torch.mean(iF, dim=2)
        iCov = GramMatrix()(input)

        tb, tc, th, tw = target.size()
        tF = target.view(tb, tc, -1)
        tMean = torch.mean(tF, dim=2)
        tCov = GramMatrix()(target)

        loss = nn.MSELoss(reduction='sum')(iMean, tMean) + nn.MSELoss(reduction='sum')(iCov, tCov)
        return loss / tb

class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        f = input.view(b, c, h * w)
        G = torch.bmm(f, f.transpose(1, 2))
        return G.div_(c * h * w)

def canny_edge(img_tensor):
    """
    img_tensor: [B, 3, H, W], 0~1 float tensor
    returns: [B, 1, H, W] float tensor of edge map
    """
    edges = []
    img_tensor = img_tensor.detach().cpu()
    for img in img_tensor:
        img_np = img.permute(1, 2, 0).numpy() * 255  # [H, W, C]
        img_np = img_np.astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian Blur to the grayscale image
        blurred = cv2.GaussianBlur(gray, (3,3), 0.5)
        
        # Apply Canny edge detector
        edge = cv2.Canny(blurred, threshold1=100, threshold2=200)
        
        edge = edge.astype(np.float32) / 255.0  # Normalize to [0, 1]
        edge = torch.from_numpy(edge).unsqueeze(0)  # [1, H, W]
        edges.append(edge)
    return torch.stack(edges).to(img_tensor.device)  # [B, 1, H, W]

class LossCriterion(nn.Module):
    def __init__(self, style_layers, content_layers, style_weight, content_weight, edge_weight=1.0):
        super(LossCriterion, self).__init__()

        self.style_layers = style_layers
        self.content_layers = content_layers
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.edge_weight = edge_weight

        self.styleLosses = [styleLoss()] * len(style_layers)
        self.contentLosses = [self.calculate_content_loss] * len(content_layers)
        self.image_filter = get_default_ssim_image_filter()


    def calculate_content_loss(self, tf_i, cf_i):
        device = tf_i.device  # tf_i의 디바이스 가져오기 (모델의 디바이스)
        tf_i = tf_i.to(device)
        cf_i = cf_i.to(device)

        # SSIM 계산을 CPU에서 하도록 강제
        tf_i_cpu = tf_i.to('cpu')
        cf_i_cpu = cf_i.to('cpu')

        input_repr = calculate_ssim_repr(tf_i_cpu, self.image_filter)
        target_repr = calculate_ssim_repr(cf_i_cpu, self.image_filter)
        ctx = calculate_simplified_ssim_ctx(cf_i_cpu, 1e-2)
        content_loss = calculate_score(input_repr, target_repr, ctx)

        content_loss = content_loss.to(device)

        return content_loss

    def forward(self, tF, sF, cF, input_img=None, output_img=None):
        # Content loss
        totalContentLoss = 0
        for i, layer in enumerate(self.content_layers):
            cf_i = cF[layer].detach()
            tf_i = tF[layer]
            loss_i = self.contentLosses[i]
            totalContentLoss += loss_i(tf_i, cf_i)
        totalContentLoss = totalContentLoss * self.content_weight

        # Style loss
        totalStyleLoss = 0
        for i, layer in enumerate(self.style_layers):
            sf_i = sF[layer].detach()
            tf_i = tF[layer]
            loss_i = self.styleLosses[i]
            totalStyleLoss += loss_i(tf_i, sf_i)
        totalStyleLoss = totalStyleLoss * self.style_weight

         # Canny edge loss
        if input_img is not None and output_img is not None:
            edge_input = canny_edge(input_img)
            edge_output = canny_edge(output_img)
            edgeLoss = F.mse_loss(edge_output, edge_input) * self.edge_weight
        else:
            edgeLoss = torch.tensor(0.0).to(tF[self.content_layers[0]].device)

        loss = totalContentLoss + totalStyleLoss + edgeLoss
        return loss, totalStyleLoss, totalContentLoss, edgeLoss
