import torch
ckpt = torch.load(r'E:\LQiu\lab_folder\Machine_learning\AGID_Project\Logs\cbnet_baseline_nobottleneck_s42\ckpt_epoch20.pth', map_location='cpu', weights_only=False)
print('keys:', list(ckpt.keys()))
mk = list(ckpt['model'].keys())
print('model keys (first 8):', mk[:8])
print('classifier.weight shape:', ckpt['model']['classifier.weight'].shape)
