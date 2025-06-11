import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

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

# Laplacian kernel
laplacian_kernel = torch.tensor([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)

def laplacian_edge(img):
    # Convert to grayscale
    gray = TF.rgb_to_grayscale(img)

    # Apply Gaussian Blur after grayscale conversion
    gaussian_blur = T.GaussianBlur(kernel_size=3, sigma=0.5)
    img_blurred = gaussian_blur(gray)

    # Apply Laplacian kernel
    kernel = laplacian_kernel.to(img.device)
    edge = F.conv2d(img_blurred, kernel, padding=1)

    return edge

class LossCriterion(nn.Module):
    def __init__(self, style_layers, content_layers, style_weight, content_weight, edge_weight=1.0):
        super(LossCriterion, self).__init__()

        self.style_layers = style_layers
        self.content_layers = content_layers
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.edge_weight = edge_weight

        self.styleLosses = [styleLoss()] * len(style_layers)
        self.contentLosses = [nn.MSELoss()] * len(content_layers)

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

        # Edge loss
        if input_img is not None and output_img is not None:
            edge_input = laplacian_edge(input_img)
            edge_output = laplacian_edge(output_img)
            edgeLoss = F.mse_loss(edge_output, edge_input) * self.edge_weight
        else:
            edgeLoss = torch.tensor(0.0).to(tF[self.content_layers[0]].device)

        loss = totalContentLoss + totalStyleLoss + edgeLoss
        return loss, totalStyleLoss, totalContentLoss, edgeLoss
