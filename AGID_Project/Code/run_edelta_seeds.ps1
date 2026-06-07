# Detached runner for the E-Delta seed-1/seed-2 retrains (P0-#2 peer-review response).
# Launched via Start-Process so it survives Claude session interrupts/resumes.
# Identical recipe to the original E-Delta (Logs/edelta_full/config.json) except --seed.
$ErrorActionPreference = 'Continue'
$env:PYTHONNOUSERSITE = 1
$env:PYTHONMALLOC = 'malloc'
Set-Location 'E:\LQiu\lab_folder\Machine_learning\AGID_Project\Code'
$py = 'E:\LQiu\conda_envs\agid\python.exe'
$common = @(
  '-m', 'cbnet_agid.train_multigen',
  '--root', '../Data/GenImage_debiased_full',
  '--generators', 'Stable_Diffusion_v1.4', 'BigGAN', 'ADM', 'Midjourney',
  '--split', 'train_25k', '--val_generators', 'all',
  '--n_concepts', '6', '--signal_channels', '512',
  '--epochs', '10', '--batch_size', '32', '--accum_steps', '2',
  '--lr_backbone', '1e-4', '--lr_head', '5e-4', '--lr_classifier', '1e-3',
  '--weight_decay', '1e-4',
  '--lambda_concept', '0.5', '--lambda_pair', '0.2', '--lambda_sparsity', '0.05',
  '--image_size', '256', '--num_workers', '2',
  '--ckpt_every', '1', '--eval_every', '1'
)
New-Item -ItemType Directory -Force 'Logs\edelta_seed1' | Out-Null
New-Item -ItemType Directory -Force 'Logs\edelta_seed2' | Out-Null

"START seed1 $(Get-Date -Format o)" | Out-File 'Logs\edelta_runner.status' -Encoding utf8
& $py @common --out_dir Logs/edelta_seed1 --seed 1 *> 'Logs\edelta_seed1\train.log'
"DONE seed1 exit=$LASTEXITCODE $(Get-Date -Format o)" | Out-File 'Logs\edelta_runner.status' -Append -Encoding utf8

& $py @common --out_dir Logs/edelta_seed2 --seed 2 *> 'Logs\edelta_seed2\train.log'
"DONE seed2 exit=$LASTEXITCODE $(Get-Date -Format o)" | Out-File 'Logs\edelta_runner.status' -Append -Encoding utf8
"ALL DONE $(Get-Date -Format o)" | Out-File 'Logs\edelta_runner.status' -Append -Encoding utf8
