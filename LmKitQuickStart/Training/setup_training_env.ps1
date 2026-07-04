# setup_training_env.ps1
# Sets up Python environment for Xprema LoRA fine-tuning
# Run once before training: .\setup_training_env.ps1

Write-Host "=== Xprema Training Environment Setup ===" -ForegroundColor Cyan

$envDir = "$PSScriptRoot\xprema_env"

# 1. Create isolated Python 3.11 env via uv (more compatible than 3.14)
Write-Host "`n[1] Creating Python 3.11 virtual environment..." -ForegroundColor Yellow
uv venv $envDir --python 3.11
if (-not $?) { Write-Host "Failed. Install uv: winget install astral-sh.uv" -ForegroundColor Red; exit 1 }

$pip = "$envDir\Scripts\pip.exe"
$py  = "$envDir\Scripts\python.exe"

# 2. Install CUDA PyTorch (CUDA 12.1 for RTX 4060)
Write-Host "`n[2] Installing PyTorch with CUDA 12.1..." -ForegroundColor Yellow
& $pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install Unsloth
Write-Host "`n[3] Installing Unsloth..." -ForegroundColor Yellow
& $pip install "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git"

# 4. Install training dependencies
Write-Host "`n[4] Installing TRL, datasets, bitsandbytes..." -ForegroundColor Yellow
& $pip install trl datasets bitsandbytes accelerate numpy

# 5. Verify CUDA
Write-Host "`n[5] Verifying CUDA..." -ForegroundColor Yellow
& $py -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

Write-Host "`n=== Setup complete ===" -ForegroundColor Green
Write-Host "To train, run:" -ForegroundColor Cyan
Write-Host "  $py $PSScriptRoot\unsloth_train.py" -ForegroundColor White
