#!/bin/bash
# Setup script for emotion classification experiments
# Run this first to verify dependencies and dataset

set -e          # Exit on error
set -o pipefail # Exit on error in any part of a pipeline

# Setup Python path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "=========================================="
echo "Emotion Classification - Setup"
echo "=========================================="

# Check Python version
echo "Checking Python version..."
python --version

# Check if required packages are installed
echo ""
echo "Checking dependencies..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchvision; print(f'Torchvision: {torchvision.__version__}')"
python -c "import timm; print(f'timm: {timm.__version__}')"
python -c "import sklearn; print('scikit-learn: OK')"
python -c "import loguru; print('loguru: OK')"
python -c "import omegaconf; print('omegaconf: OK')"
python -c "from tensorboard import version; print(f'tensorboard: OK')"

# Check CUDA availability
echo ""
echo "Checking CUDA..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); \
           print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); \
           print(f'GPU count: {torch.cuda.device_count()}'); \
           print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Check dataset
echo ""
echo "Checking Emoset dataset..."
DATASET_DIR="${EMOSET_ROOT:-datasets/emoset}"
echo "Looking for dataset at: $DATASET_DIR"

if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset directory not found at $DATASET_DIR"
    echo "Please set EMOSET_ROOT environment variable or place dataset at datasets/emoset/"
    exit 1
fi

# Check required files
REQUIRED_FILES=("train.json" "val.json" "test.json" "info.json")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$DATASET_DIR/$file" ]; then
        echo "ERROR: Missing required file: $file"
        exit 1
    fi
    echo "✓ Found $file"
done

# Count samples
echo ""
echo "Dataset statistics:"
python -c "
import json
import sys
sys.path.insert(0, 'emoset')
dataset_dir = '$DATASET_DIR'

for split in ['train', 'val', 'test']:
    with open(f'{dataset_dir}/{split}.json', 'r') as f:
        data = json.load(f)
        print(f'{split:5s}: {len(data):5d} samples')
"

# Create output directories
echo ""
echo "Creating output directories..."
mkdir -p output/zero_shot
mkdir -p output/clip_coop/checkpoints
mkdir -p output/meru_coop/checkpoints
mkdir -p logs
echo "✓ Output directories created"

# Test dataset loading
echo ""
echo "Testing dataset loading..."
python -c "
import sys
sys.path.insert(0, 'emoset')
from Emoset import EmoSet

dataset = EmoSet(
    data_root='$DATASET_DIR',
    num_emotion_classes=8,
    phase='train'
)
print(f'✓ Successfully loaded {len(dataset)} training samples')
sample = dataset[0]
print(f'✓ Sample keys: {list(sample.keys())}')
print(f'✓ Image shape: {sample[\"image\"].shape}')
print(f'✓ Emotion label: {sample[\"emotion_label_idx\"]}')
"

# Test emotion module import
echo ""
echo "Testing emotion module..."
python -c "
from meru.emotion import CLIPCoOpEmotion, MERUCoOpEmotion, register_emoset
print('✓ Emotion module imported successfully')
emotion_names = register_emoset()
print(f'✓ Registered emotions: {emotion_names}')
"

echo ""
echo "=========================================="
echo "Setup Complete! ✓"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run zero-shot evaluation: bash scripts/emotion_experiments/01_zero_shot.sh"
echo "2. Train CLIP+CoOp: bash scripts/emotion_experiments/02_train_clip_coop.sh"
echo "3. Train MERU+CoOp: bash scripts/emotion_experiments/03_train_meru_coop.sh"
echo ""
echo "Note: Make sure you have pretrained CLIP/MERU checkpoints before proceeding."
