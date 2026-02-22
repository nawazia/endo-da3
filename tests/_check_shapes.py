import torch
from endo_da3 import load_da3_dino

ckpt = torch.load('/home/in4218/code/GastroNet/gastronet/dinov2.pth', map_location='cpu')
gastro_sd = {k.replace('backbone.', ''): v for k, v in ckpt['teacher'].items() if k.startswith('backbone.')}

model = load_da3_dino(device='cpu')
model_sd = {k: p for k, p in model.named_parameters()}
model_buf = {k: b for k, b in model.named_buffers()}
model_all = {**model_sd, **model_buf}

print('Shape mismatches:')
for k, v in gastro_sd.items():
    if k in model_all and v.shape != model_all[k].shape:
        print(f'  {k}: gastronet={tuple(v.shape)}  da3={tuple(model_all[k].shape)}')

print('Keys in gastronet not in DA3 model:')
for k in gastro_sd:
    if k not in model_all:
        print(f'  {k}')
