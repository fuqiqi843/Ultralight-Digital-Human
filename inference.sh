#!/bin/bash

# ========= 参数配置 =========
WAV_PATH=${1:-"input/test.wav"}
AUDIO_FEAT=${2:-"input/test_hu.npy"}
DATASET_DIR=${3:-"./input"}
CHECKPOINT=${4:-"output/checkpoint/195.pth"}
TMP_VIDEO=${5:-"output/temp_test.mp4"}
FINAL_VIDEO=${6:-"output/result_test.mp4"}
ASR_TYPE=${7:-"hubert"}  # ASR 类型，默认是 hubert


# Step 1: 提取音频特征（根据 ASR_TYPE）1111111111
echo "[INFO] 使用 $ASR_TYPE 提取音频特征..."
if [ "$ASR_TYPE" = "hubert" ]; then
    python data_utils/hubert.py --wav "$WAV_PATH"
elif [ "$ASR_TYPE" = "wenet" ]; then
    echo "[INFO] 切换到 data_utils 目录以确保配置文件路径正确..."
    (cd data_utils && python wenet_infer.py "../$WAV_PATH")
else
    echo "[ERROR] 不支持的 ASR 类型: $ASR_TYPE"
    exit 1
fi

# 打印即将执行的命令
echo "python inference.py \\"
echo "    --asr \"$ASR_TYPE\" \\"
echo "    --dataset \"$DATASET_DIR\" \\"
echo "    --audio_feat \"$AUDIO_FEAT\" \\"
echo "    --save_path \"$TMP_VIDEO\" \\"
echo "    --checkpoint \"$CHECKPOINT\""
# Step 2: 执行推理，生成视频（不带音频）
echo "运行推理中..."
python inference.py \
    --asr "$ASR_TYPE" \
    --dataset "$DATASET_DIR" \
    --audio_feat "$AUDIO_FEAT" \
    --save_path "$TMP_VIDEO" \
    --checkpoint "$CHECKPOINT"

# Step 3: 使用 FFmpeg 合成音频和视频
echo "合成最终视频中..."
ffmpeg -i "$TMP_VIDEO" -i "$WAV_PATH" -c:v libx264 -c:a aac "$FINAL_VIDEO"

echo "完成！输出文件：$FINAL_VIDEO"
