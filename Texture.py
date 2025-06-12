import os
import io
import sys
from string import Template
from pathlib import Path

import numpy as np
import PIL.Image
import matplotlib.pylab as plt



from IPython.display import clear_output, display, Image, HTML


import OpenGL.GL as gl

from lucid.misc.gl.glcontext import create_opengl_context, destroy_opengl_context
from lucid.misc.gl import meshutil
from lucid.misc.gl import glrenderer
import lucid.misc.io.showing as show
import lucid.misc.io as lucid_io
from lucid.misc.tfutil import create_session

from lucid.modelzoo import vision_models
from lucid.optvis import objectives
from lucid.optvis import param
from lucid.optvis.style import StyleLoss, mean_l1_loss
from lucid.optvis.param.spatial import sample_bilinear
os.environ['PYOPENGL_PLATFORM'] = 'glfw'


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


######################style transfer####################################
import os
import torch
import argparse
from libs.MatrixTest_r import MulLayer
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from libs.utils import print_options
from libs.models import encoder3,encoder4, encoder5
from libs.models import decoder3,decoder4, decoder5
from libs.LoaderPhotoReal import Dataset


parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='models/vgg_r41.pth',
                    help='pre-trained encoder path')
parser.add_argument("--decoder_dir", default='models/dec_r41.pth',
                    help='pre-trained decoder path')
parser.add_argument("--matrixPath", default='trainingOutput/ssim/r41.pth',
                    help='pre-trained model path')
parser.add_argument("--stylePath", default="data/photo_real/style/images/",
                    help='path to style image')
parser.add_argument("--styleSegPath", default="data/photo_real/styleSeg/",
                    help='path to style image masks')
parser.add_argument("--contentPath", default="data/photo_real/content/images/",
                    help='path to content image')
parser.add_argument("--contentSegPath", default="data/photo_real/contentSeg/",
                    help='path to content image masks')
parser.add_argument("--outf", default="Artistic/canny/",
                    help='path to transferred images')
parser.add_argument("--batchSize", type=int,default=1,
                    help='batch size')
parser.add_argument('--loadSize', type=int, default=512,
                    help='scale image size')
parser.add_argument('--fineSize', type=int, default=512,
                    help='crop image size')
parser.add_argument("--layer", default="r41",
                    help='which features to transfer, either r31 or r41')



################# PREPARATIONS #################
opt = parser.parse_args()
opt.cuda = torch.cuda.is_available()
print_options(opt)

os.makedirs(opt.outf,exist_ok=True)
cudnn.benchmark = True

################# DATA #################
dataset = Dataset(opt.contentPath,opt.stylePath,opt.contentSegPath,opt.styleSegPath,opt.fineSize)
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=1,
                                     shuffle=False)

################# GLOBAL VARIABLE #################
contentV = torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
styleV = torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize)

################# MODEL #################
if(opt.layer == 'r31'):
    vgg = encoder3()
    dec = decoder3()
elif(opt.layer == 'r41'):
    vgg = encoder4()
    dec = decoder4()
matrix = MulLayer(opt.layer)
vgg.load_state_dict(torch.load(opt.vgg_dir))
dec.load_state_dict(torch.load(opt.decoder_dir))
matrix.load_state_dict(torch.load(opt.matrixPath, map_location=torch.device('cpu')))


#create context
model = vision_models.InceptionV1()
model.load_graphdef()
create_opengl_context()
print(gl.glGetString(gl.GL_VERSION))
renderer = glrenderer.MeshRenderer((512,512))

#masking
def ExtractMask(Seg):
    color_codes = ['blue', 'green', 'black', 'white', 'red', 'yellow', 'grey', 'lightblue', 'purple']
    masks = []
    for color in color_codes:
        mask = MaskHelper(Seg,color)
        masks.append(mask)
    return masks

def MaskHelper(seg,color):
    # green
    mask = torch.Tensor()
    if(color == 'green'):
        mask = torch.lt(seg[0],0.1)
        mask = torch.mul(mask,torch.gt(seg[1],1-0.1))
        mask = torch.mul(mask,torch.lt(seg[2],0.1))
    elif(color == 'black'):
        mask = torch.lt(seg[0], 0.1)
        mask = torch.mul(mask,torch.lt(seg[1], 0.1))
        mask = torch.mul(mask,torch.lt(seg[2], 0.1))
    elif(color == 'white'):
        mask = torch.gt(seg[0], 1-0.1)
        mask = torch.mul(mask,torch.gt(seg[1], 1-0.1))
        mask = torch.mul(mask,torch.gt(seg[2], 1-0.1))
    elif(color == 'red'):
        mask = torch.gt(seg[0], 1-0.1)
        mask = torch.mul(mask,torch.lt(seg[1], 0.1))
        mask = torch.mul(mask,torch.lt(seg[2], 0.1))
    elif(color == 'blue'):
        mask = torch.lt(seg[0], 0.1)
        mask = torch.mul(mask,torch.lt(seg[1], 0.1))
        mask = torch.mul(mask,torch.gt(seg[2], 1-0.1))
    elif(color == 'yellow'):
        mask = torch.gt(seg[0], 1-0.1)
        mask = torch.mul(mask,torch.gt(seg[1], 1-0.1))
        mask = torch.mul(mask,torch.lt(seg[2], 0.1))
    elif(color == 'grey'):
        mask = torch.lt(seg[0], 0.1)
        mask = torch.lt(seg[0], 0.5)  # R < 0.5
        mask = torch.mul(mask, torch.gt(seg[1], 0.1))  # G > 0.1
        mask = torch.mul(mask, torch.lt(seg[1], 0.5))  # G < 0.5
        mask = torch.mul(mask, torch.gt(seg[2], 0.1))  # B > 0.1
        mask = torch.mul(mask, torch.lt(seg[2], 0.5))
    elif(color == 'lightblue'):
        mask = torch.lt(seg[0], 0.1)
        mask = torch.mul(mask,torch.gt(seg[1], 1-0.1))
        mask = torch.mul(mask,torch.gt(seg[2], 1-0.1))
    elif(color == 'purple'):
        mask = torch.gt(seg[0], 1-0.1)
        mask = torch.mul(mask,torch.lt(seg[1], 0.1))
        mask = torch.mul(mask,torch.gt(seg[2], 1-0.1))
    else:
        print('MaskHelper(): color not recognized, color = ' + color)
    return mask.float()

#prepare data
def prepare_image(fn, size=None):
  #data = lucid_io.reading.read(fn)
  im = PIL.Image.open(fn).convert('RGB')
  if size:
    im = im.resize(size, PIL.Image.LANCZOS)
    im_np = np.float32(im) / 255.0  # 0-1 범위로 정규화
    return im_np
  
def overlay_non_black(temp_image, original_image):
    temp_np = np.array(temp_image).astype(np.float32) / 255.0
    orig_np = np.array(original_image)
    
    mask = np.any(temp_np != [0, 0, 0], axis=-1)  #
    

    blended_np = orig_np.copy()
    blended_np[mask] = temp_np[mask]
    
    return blended_np

TEXTURE_SIZE = 512
mesh = meshutil.load_obj('mesh/blub/blub.obj')
mesh = meshutil.normalize_mesh(mesh)
original_texture = prepare_image('mesh/blub/noise_blub.png', (TEXTURE_SIZE, TEXTURE_SIZE))
style='mesh/s1.jpg'


#create session
sess = create_session(timeout_sec=0)

#texture create
NUM_VIEWS = 20
k=0
tex_res=512
new_texture = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, 3), dtype=np.float32)
alpha_accumulator = np.zeros((TEXTURE_SIZE, TEXTURE_SIZE, 1), dtype=np.float32)


import os

# 경로를 변수로 정의
view_dir = "views/blub_8_1"
inter_dir = "inter/blub_8_1"
final_dir = "final/blub_8_1"

# 각 경로가 존재하지 않으면 생성
os.makedirs(view_dir, exist_ok=True)
os.makedirs(inter_dir, exist_ok=True)
os.makedirs(final_dir, exist_ok=True)

import time
start_time = time.time()

for i in range(NUM_VIEWS):
    print(k)
    view = meshutil.sample_view(8, 8)
    fragments = renderer.render_mesh(
        modelview=view,
        position=mesh['position'], uv=mesh['uv'], face=mesh['face']
    )
    
    t_uv = fragments[..., :2]
    t_alpha = fragments[..., 3:]
    t_sampled_texture = sample_bilinear(original_texture, t_uv)

    final_image = t_sampled_texture * t_alpha


    white_texture = tf.ones_like(t_sampled_texture)
    mask_image = white_texture * t_alpha

    rendered_image_nps = sess.run(final_image)


    mask_image_nps = sess.run(mask_image)


    content = torch.tensor(rendered_image_nps, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
    content = content.to('cpu')

    content=rendered_image_nps
    cmask = mask_image_nps

    content = torch.tensor(content, dtype=torch.float32)
    cmask = torch.tensor(cmask, dtype=torch.float32)
    contentV.resize_(content.size()).copy_(content)
    cmask.resize_(cmask.size()).copy_(cmask)

    if contentV.dim() == 3:
        contentV = contentV.unsqueeze(0)  

    if cmask.dim() == 3:
        cmask = cmask.unsqueeze(0)  
        

    contentV = contentV.permute(0, 3, 1, 2)
    style_image = PIL.Image.open(style).convert('RGB')

    from torchvision import transforms
    transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((512, 512))])
        
    styleV = transform(style_image).unsqueeze(0)
    styleV.resize_(styleV.size()).copy_(styleV)


    #extract mask
    mask_image_np = np.clip(mask_image_nps * 255, 0, 255).astype(np.uint8)
    masking = PIL.Image.fromarray(mask_image_np).convert('RGB')
    masking = transform(masking)
    cmasksss = ExtractMask(masking)

    cmasksss = ExtractMask(masking)
    smask= PIL.Image.new('RGB', (512,512), color=(255, 255, 255))
    smask=transform(smask)
    smasks=ExtractMask(smask)
    
    with torch.no_grad():
        cF = vgg(contentV)
        sF = vgg(styleV)
        if(opt.layer == 'r41'):
            feature= matrix(cF[opt.layer], sF[opt.layer], cmasksss, smasks)
        else:
            feature= matrix(cF, sF, cmask)
        transfer = dec(feature).clamp(0, 1)

        transfer_image = transfer.squeeze(0).permute(1, 2, 0).cpu().numpy()
        transfer_image = np.clip(transfer_image * 255, 0, 255).astype(np.uint8)
        PIL.Image.fromarray(transfer_image).save(f"{view_dir}/transfer_output_{k:04d}.png")

    height, width = transfer_image.shape[:2]
    for y in range(height):
        for x in range(width):
            uv = fragments[y, x, :2]
            alpha = fragments[y, x, 3:4]

            if alpha < 0.1:
                continue

            fu = uv[0] * (TEXTURE_SIZE - 1)
            fv = (1-uv[1]) * (TEXTURE_SIZE - 1)

            u0 = int(np.floor(fu))
            v0 = int(np.floor(fv))
            u1 = u0 + 1
            v1 = v0 + 1

            du = fu - u0
            dv = fv - v0

            w00 = (1 - du) * (1 - dv)
            w01 = (1 - du) * dv
            w10 = du * (1 - dv)
            w11 = du * dv

            for (uu, vv, w) in [(u0, v0, w00), (u0, v1, w01), (u1, v0, w10), (u1, v1, w11)]:
                if 0 <= uu < TEXTURE_SIZE and 0 <= vv < TEXTURE_SIZE:
                    new_texture[vv,uu] += transfer_image[y, x] * alpha * w
                    alpha_accumulator[vv,uu] += alpha * w

    temp_alpha = np.clip(alpha_accumulator, 1e-5, None)
    temp_texture = (new_texture / temp_alpha).astype(np.uint8)
    temp_texture_image = PIL.Image.fromarray(temp_texture)
    temp_texture_image.save(f"{inter_dir}/intermediate_texture_{k:04d}.png")
    original_texture=overlay_non_black(temp_texture_image, original_texture)

    k=k+1

alpha_accumulator = np.clip(alpha_accumulator, 1e-5, None)  
final_texture = (new_texture / alpha_accumulator).astype(np.uint8)
final_texture_image = PIL.Image.fromarray(final_texture.astype(np.uint8))
final_texture_image.save(f"{final_dir}/final_texture.png")

end_time = time.time()
print("총 실행 시간: {:.2f}초".format(end_time - start_time))