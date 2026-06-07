"""Quick smoke-test of the training pipeline: imports, forward, loss."""
import torch
from cbnet_agid.models import CBNetAGID
from cbnet_agid.losses import CBNetLoss
from cbnet_agid.data.genimage import GenImageDataset
from cbnet_agid.data.transforms import get_train_transform

print("1. imports OK")

model = CBNetAGID(n_concepts=6, pretrained=False)
x = torch.randn(4, 3, 256, 256)
out = model(x)
print(f"2. forward OK: logit={out['logit'].shape}, prob={out['prob'].shape}, concepts={out['concepts'].shape}")

loss_fn = CBNetLoss(lambda_concept=0.5, lambda_gen=0.0, lambda_sparsity=0.05)
target_concepts = torch.rand(4, 6)
losses = loss_fn(out, torch.tensor([0, 1, 0, 1]).float(), target_concepts=target_concepts)
for k, v in losses.items():
    print(f"3. {k}: {v.item():.4f}")

print("4. ALL CHECKS PASSED")
