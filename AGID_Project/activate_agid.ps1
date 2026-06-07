# Activate the agid environment with proper isolation.
# Usage: . .\activate_agid.ps1   (note the leading dot + space to source it)
# This dot-sources the script so env-var changes persist in your current shell.

$env:PYTHONNOUSERSITE = "1"
$env:CONDA_ENV_PATH = "E:\LQiu\conda_envs\agid"

# Add agid env to PATH (Python + Scripts)
$env:PATH = "$env:CONDA_ENV_PATH;$env:CONDA_ENV_PATH\Scripts;$env:CONDA_ENV_PATH\Library\bin;" + $env:PATH

# Verify
Write-Host "agid env activated."
Write-Host "  Python:         $(& "$env:CONDA_ENV_PATH\python.exe" --version 2>&1)"
Write-Host "  PYTHONNOUSERSITE: $env:PYTHONNOUSERSITE"
Write-Host ""
Write-Host "Quick sanity check:"
& "$env:CONDA_ENV_PATH\python.exe" -c "import torch; print('  torch ' + torch.__version__ + ' cuda=' + str(torch.cuda.is_available()) + ' device=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'))"
