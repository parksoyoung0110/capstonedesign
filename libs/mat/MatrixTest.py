import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torch.autograd import Variable
import torchvision.utils as vutils

class CNN2(nn.Module):
    def __init__(self,layer,matrixSize=32):
        super(CNN2,self).__init__()
        # 256x64x64
        if(layer == 'r31'):
            self.convs = nn.Sequential(nn.Conv2d(256,128,3,1,1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128,64,3,1,1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64,matrixSize,3,1,1))
        elif(layer == 'r41'):
            # 512x32x32
            self.convs = nn.Sequential(nn.Conv2d(512,256,3,1,1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256,128,3,1,1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128,matrixSize,3,1,1))
        self.fc = nn.Linear(32*32,32*32)

    def forward(self,x):
        out = self.convs(x)
        # 32x8x8
        b,c,h,w = out.size()
        out = out.view(b,c,-1)
        # 32x64
        out = torch.bmm(out,out.transpose(1,2)).div(h*w)
        # 32x32
        out = out.view(out.size(0),-1)
        return self.fc(out)

class CNN(nn.Module):
    def __init__(self,layer,matrixSize=32):
        super(CNN,self).__init__()
        # 256x64x64
        if(layer == 'r31'):
            self.convs = nn.Sequential(nn.Conv2d(256,128,3,1,1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128,64,3,1,1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(64,matrixSize,3,1,1))
        elif(layer == 'r41'):
            # 512x32x32
            self.convs = nn.Sequential(nn.Conv2d(512,256,3,1,1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256,128,3,1,1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128,matrixSize,3,1,1))
        self.fc = nn.Linear(32*32,32*32)

    def forward(self,x,masks,style=False):
        color_code_number = 9
        xb,xc,xh,xw = x.size()
        x = x.view(xc,-1)
        feature_sub_mean = x.clone()
        for i in range(color_code_number):
            mask = masks[i].clone().squeeze(0)
            mask = cv2.resize(mask.numpy(),(xw,xh),interpolation=cv2.INTER_NEAREST)
            mask = torch.FloatTensor(mask)
            mask = mask.long()
            if(torch.sum(mask) >= 10):
                mask = mask.view(-1)

                # dilation here
                """
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
                mask = mask.cpu().numpy()
                mask = cv2.dilate(mask.astype(np.float32), kernel)
                mask = torch.from_numpy(mask)
                mask = mask.squeeze()
                """

                fgmask = (mask>0).nonzero().squeeze(1)
                fgmask = fgmask.to('cpu')
                selectFeature = torch.index_select(x,1,fgmask) # 32x96
                # subtract mean
                f_mean = torch.mean(selectFeature,1)
                f_mean = f_mean.unsqueeze(1).expand_as(selectFeature)
                selectFeature = selectFeature - f_mean
                feature_sub_mean.index_copy_(1,fgmask,selectFeature)

        feature = self.convs(feature_sub_mean.view(xb,xc,xh,xw))
        # 32x16x16
        b,c,h,w = feature.size()
        transMatrices = {}
        feature = feature.view(c,-1)

        for i in range(color_code_number):
            mask = masks[i].clone().squeeze(0)
            mask = cv2.resize(mask.numpy(),(w,h),interpolation=cv2.INTER_NEAREST)
            mask = torch.FloatTensor(mask)
            mask = mask.long()
            if(torch.sum(mask) >= 10):
                mask = mask.view(-1)
                fgmask = Variable((mask==1).nonzero().squeeze(1))
                fgmask = fgmask.to('cpu')
                selectFeature = torch.index_select(feature,1,fgmask) # 32x96
                tc,tN = selectFeature.size()

                covMatrix = torch.mm(selectFeature,selectFeature.transpose(0,1)).div(tN)
                transmatrix = self.fc(covMatrix.view(-1))
                transMatrices[i] = transmatrix
        return transMatrices,feature_sub_mean

class MulLayer(nn.Module):
    def __init__(self,layer,matrixSize=32):
        super(MulLayer,self).__init__()
        self.snet = CNN2(layer)
        self.cnet = CNN(layer)
        self.matrixSize = matrixSize

        if(layer == 'r41'):
            self.compress = nn.Conv2d(512,matrixSize,1,1,0)
            self.unzip = nn.Conv2d(matrixSize,512,1,1,0)
        elif(layer == 'r31'):
            self.compress = nn.Conv2d(256,matrixSize,1,1,0)
            self.unzip = nn.Conv2d(matrixSize,256,1,1,0)

    def forward(self,cF,sF,cmasks):

        sb,sc,sh,sw = sF.size()
        sb,sc,sh,sw = sF.size()#
        sFF = sF.view(sb,sc,-1)
        sMean = torch.mean(sFF,dim=2,keepdim=True)
        sMean = sMean.unsqueeze(3)
        sMeanC = sMean.expand_as(cF)
        


        sMatrices= self.snet(sF)
        cMatrices,cF_sub_mean = self.cnet(cF,cmasks,style=False)

        compress_content = self.compress(cF_sub_mean.view(cF.size()))
        cb,cc,ch,cw = compress_content.size()
        compress_content = compress_content.view(cc,-1)
        transfeature = compress_content.clone()
        color_code_number = 9
        finalSMean = Variable(torch.zeros(cF.size()).to('cpu'))
        finalSMean = finalSMean.view(sc,-1)
        for i in range(color_code_number):
            cmask = cmasks[i].clone().squeeze(0)

            cmask = cv2.resize(cmask.numpy(),(cw,ch),interpolation=cv2.INTER_NEAREST)
            cmask = torch.FloatTensor(cmask)
            cmask = cmask.long()

            if(torch.sum(cmask) >= 10 and (i in cMatrices)):
                cmask = cmask.view(-1)
                fgcmask = Variable((cmask==1).nonzero().squeeze(1))
                fgcmask = fgcmask.to('cpu')

                sMatrix = self.snet(sF)#
                cMatrix = cMatrices[i]

                sMatrix = sMatrix.view(self.matrixSize, self.matrixSize)
                cMatrix = cMatrix.view(self.matrixSize,self.matrixSize)

                transmatrix = torch.mm(sMatrix,cMatrix) # (C*C)

                compress_content_select = torch.index_select(compress_content,1,fgcmask)

                transfeatureFG = torch.mm(transmatrix,compress_content_select)
                transfeature.index_copy_(1,fgcmask,transfeatureFG)

            # Unzip the transformed feature and add the final style mean
            out = self.unzip(transfeature.view(cb, cc, ch, cw))
            return out + finalSMean.view(out.size())
