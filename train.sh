#!/bin/bash

# ========= 超参数配置 =========
VIDEO_PATH=${1:-"./input/sample.mp4"}           # 视频路径
DATASET_DIR=${2:-"./data_utils/data"}           # 数据集存放路径
SYNCNET_CKPT_DIR=${3:-"./output/syncnet_ckpt"}  # SyncNet 预训练模型路径
MAIN_CKPT_DIR=${4:-"./output/checkpoint"}       # 主模型保存路径
ASR_TYPE=${5:-"hubert"}                         # 音频编码器类型
SYNCNET_BATCH_SIZE=${6:-16}                     # SyncNet模型的batch_size
MAIN_BATCH_SIZE=${7:-1}                         # 主模型的batch_size
MODEL_EPOCHS=${8:-200}                          # 训练轮数
SYNCNET_EPOCHS=${9:-40}                         # SyncNet 训练轮数
LEARNING_RATE=${10:-0.01}                     # 学习率

# ========= 步骤 1：数据预处理 =========
echo "[INFO] 开始数据预处理..."
cd data_utils
python process.py "$VIDEO_PATH" --asr "$ASR_TYPE"
cd ..

# ========= 步骤 2：训练 SyncNet =========
echo "[INFO] 开始训练 SyncNet..."
echo "[CMD] python syncnet.py --save_dir \"$SYNCNET_CKPT_DIR\" --dataset_dir \"$DATASET_DIR\" --asr \"$ASR_TYPE\" --epochs \"$SYNCNET_EPOCHS\" --batch_size \"$SYNCNET_BATCH_SIZE\""
python syncnet.py \
    --save_dir "$SYNCNET_CKPT_DIR" \
    --dataset_dir "$DATASET_DIR" \
    --asr "$ASR_TYPE" \
    --epochs "$SYNCNET_EPOCHS" \
    --batch_size "$SYNCNET_BATCH_SIZE"

# ========= 自动选择 SyncNet 最优模型（最小 loss） =========
echo "[INFO] 正在选择 SyncNet 最优模型（最小 loss）..."
BEST_SYNCNET_CKPT=$(ls "$SYNCNET_CKPT_DIR"/*.pth | sed -E 's/.*_loss_([0-9.]+)\.pth/\1 \0/' | sort -n | head -n 1 | cut -d' ' -f2)

if [ -z "$BEST_SYNCNET_CKPT" ]; then
    echo "[ERROR] 未找到包含 loss 信息的 SyncNet checkpoint 文件！"
    exit 1
fi

# ========= 步骤 3：训练主模型 =========
echo "[INFO] 开始训练主模型..."

if [ "$ASR_TYPE" = "hubert" ]; then
    echo "[INFO] 使用 SyncNet 和 hubert 进行训练..."
    python train.py \
        --dataset_dir "$DATASET_DIR" \
        --save_dir "$MAIN_CKPT_DIR" \
        --asr "$ASR_TYPE" \
        --use_syncnet \
        --syncnet_checkpoint "$BEST_SYNCNET_CKPT" \
        --epochs "$MODEL_EPOCHS" \
        --batchsize "$MAIN_BATCH_SIZE" \
        --lr "$LEARNING_RATE"
else
    echo "[INFO] 不使用 SyncNet，仅训练主模型..."
    python train.py \
        --dataset_dir "$DATASET_DIR" \
        --save_dir "$MAIN_CKPT_DIR" \
        --asr "$ASR_TYPE" \
        --epochs "$MODEL_EPOCHS" \
        --batchsize "$MAIN_BATCH_SIZE" \
        --lr "$LEARNING_RATE"
fi

