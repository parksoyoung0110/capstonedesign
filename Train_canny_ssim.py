import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from libs.Loader import Dataset
from libs.Matrix import MulLayer
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from libs.utils import print_options
from libs.Criterion_canny_ssim import LossCriterion
from libs.models import encoder3, encoder4
from libs.models import decoder3, decoder4
from libs.models import encoder5 as loss_network

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vgg_dir", default='models/vgg_r41.pth')
    parser.add_argument("--loss_network_dir", default='models/vgg_r51.pth')
    parser.add_argument("--decoder_dir", default='models/dec_r41.pth')
    parser.add_argument("--stylePath", default="train")
    parser.add_argument("--contentPath", default="train2014")
    parser.add_argument("--outf", default="trainingOutput/ssim_20")
    parser.add_argument("--content_layers", default="r41")
    parser.add_argument("--style_layers", default="r11,r21,r31,r41")
    parser.add_argument("--batchSize", type=int, default=8)
    parser.add_argument("--niter", type=int, default=10000)
    parser.add_argument('--loadSize', type=int, default=300)
    parser.add_argument('--fineSize', type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--content_weight", type=float, default=20.0)
    parser.add_argument("--style_weight", type=float, default=0.02)
    parser.add_argument("--edge_weight", type=float, default=4)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--save_interval", type=int, default=5000)
    parser.add_argument("--layer", default="r41")

    opt = parser.parse_args()
    opt.content_layers = opt.content_layers.split(',')
    opt.style_layers = opt.style_layers.split(',')
    opt.cuda = torch.cuda.is_available()
    if opt.cuda:
        torch.cuda.set_device(opt.gpu_id)

    os.makedirs(opt.outf, exist_ok=True)
    cudnn.benchmark = True
    print_options(opt)

    # Data
    content_dataset = Dataset(opt.contentPath, opt.loadSize, opt.fineSize)
    style_dataset = Dataset(opt.stylePath, opt.loadSize, opt.fineSize)
    content_loader_ = torch.utils.data.DataLoader(content_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=1, drop_last=True)
    style_loader_ = torch.utils.data.DataLoader(style_dataset, batch_size=opt.batchSize, shuffle=True, num_workers=1, drop_last=True)
    content_loader = iter(content_loader_)
    style_loader = iter(style_loader_)

    # Model
    vgg5 = loss_network()
    if opt.layer == 'r31':
        matrix = MulLayer('r31')
        vgg = encoder3()
        dec = decoder3()
    else:
        matrix = MulLayer('r41')
        vgg = encoder4()
        dec = decoder4()
    vgg.load_state_dict(torch.load(opt.vgg_dir))
    dec.load_state_dict(torch.load(opt.decoder_dir))
    vgg5.load_state_dict(torch.load(opt.loss_network_dir))

    for param in vgg.parameters(): param.requires_grad = False
    for param in vgg5.parameters(): param.requires_grad = False
    for param in dec.parameters(): param.requires_grad = False

    criterion = LossCriterion(opt.style_layers, opt.content_layers, opt.style_weight, opt.content_weight, opt.edge_weight)
    optimizer = optim.Adam(matrix.parameters(), opt.lr)

    contentV = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)
    styleV = torch.Tensor(opt.batchSize, 3, opt.fineSize, opt.fineSize)

    if opt.cuda:
        vgg.cuda(), dec.cuda(), vgg5.cuda(), matrix.cuda()
        contentV, styleV = contentV.cuda(), styleV.cuda()

    def adjust_learning_rate(optimizer, iteration):
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.lr / (1 + iteration * 1e-5)

    log_file_path = os.path.join(opt.outf, "train_log.txt")
    with open(log_file_path, "a") as log_file:
        for iteration in range(1, opt.niter + 1):
            optimizer.zero_grad()
            try:
                content, _ = next(content_loader)
            except:
                content_loader = iter(content_loader_)
                content, _ = next(content_loader)

            try:
                style, _ = next(style_loader)
            except:
                style_loader = iter(style_loader_)
                style, _ = next(style_loader)

            contentV.resize_(content.size()).copy_(content)
            styleV.resize_(style.size()).copy_(style)

            sF = vgg(styleV)
            cF = vgg(contentV)

            if opt.layer == 'r41':
                feature, transmatrix = matrix(cF[opt.layer], sF[opt.layer])
            else:
                feature, transmatrix = matrix(cF, sF)

            transfer = dec(feature)

            sF_loss = vgg5(styleV)
            cF_loss = vgg5(contentV)
            tF = vgg5(transfer)
            loss, styleLoss, contentLoss, edgeLoss = criterion(tF, sF_loss, cF_loss, input_img=contentV, output_img=transfer)

            loss.backward()
            optimizer.step()

            print('Iter: [%d/%d] Loss: %.4f content: %.4f style: %.4f edge: %.4f LR: %.6f' %
                (opt.niter, iteration, loss, contentLoss, styleLoss, edgeLoss, optimizer.param_groups[0]['lr']))
            
            log_msg = 'Iter: [%d/%d] Loss: %.4f content: %.4f style: %.4f edge: %.4f LR: %.6f' % (
                opt.niter, iteration, loss, contentLoss, styleLoss, edgeLoss, optimizer.param_groups[0]['lr']
            )
            
            log_file.write(log_msg + '\n')
            log_file.flush()

            adjust_learning_rate(optimizer, iteration)

            if iteration % opt.log_interval == 0:
                transfer = transfer.clamp(0, 1)
                concat = torch.cat((content, style, transfer.cpu()), dim=0)
                vutils.save_image(concat, '%s/%d.png' % (opt.outf, iteration), normalize=True, scale_each=True, nrow=opt.batchSize)

            if iteration % opt.save_interval == 0:
                torch.save(matrix.state_dict(), '%s/%s.pth' % (opt.outf, opt.layer))

if __name__ == '__main__':
    main()