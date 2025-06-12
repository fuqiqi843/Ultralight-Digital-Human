import time

import torch
from torch import nn
from torch.ao.quantization import QuantStub, DeQuantStub
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.quantization
import cv2
import os
import numpy as np
from torch import optim
import random
import argparse



class Dataset(object):
    def __init__(self, dataset_dir, mode):
        
        self.img_path_list = []
        self.lms_path_list = []
        
        for i in range(len(os.listdir(dataset_dir+"/full_body_img/"))):

            img_path = os.path.join(dataset_dir+"/full_body_img/", str(i)+".jpg")
            lms_path = os.path.join(dataset_dir+"/landmarks/", str(i)+".lms")
            self.img_path_list.append(img_path)
            self.lms_path_list.append(lms_path)
                
        if mode=="wenet":
            audio_feats_path = dataset_dir+"/aud_wenet.npy"
        if mode=="hubert":
            audio_feats_path = dataset_dir+"/aud_hu.npy"
        self.mode = mode
        self.audio_feats = np.load(audio_feats_path)
        self.audio_feats = self.audio_feats.astype(np.float32)
        
    def __len__(self):

        return self.audio_feats.shape[0]-1

    def get_audio_features(self, features, index):
        
        left = index - 8
        right = index + 8
        pad_left = 0
        pad_right = 0
        if left < 0:
            pad_left = -left
            left = 0
        if right > features.shape[0]:
            pad_right = right - features.shape[0]
            right = features.shape[0]
        auds = torch.from_numpy(features[left:right])
        if pad_left > 0:
            auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
        if pad_right > 0:
            auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0) # [8, 16]
        return auds
    
    def process_img(self, img, lms_path, img_ex, lms_path_ex):

        lms_list = []
        with open(lms_path, "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                arr = line.split(" ")
                arr = np.array(arr, dtype=np.float32)
                lms_list.append(arr)
        lms = np.array(lms_list, dtype=np.int32)
        xmin = lms[1][0]
        ymin = lms[52][1]
        
        xmax = lms[31][0]
        width = xmax - xmin
        ymax = ymin + width
        
        crop_img = img[ymin:ymax, xmin:xmax]
        crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
        img_real = crop_img[4:164, 4:164].copy()
        img_real_ori = img_real.copy()
        img_real_ori = img_real_ori.transpose(2,0,1).astype(np.float32)
        img_real_T = torch.from_numpy(img_real_ori / 255.0)
        
        return img_real_T

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path_list[idx])
        lms_path = self.lms_path_list[idx]
        
        ex_int = random.randint(0, self.__len__()-1)
        img_ex = cv2.imread(self.img_path_list[ex_int])
        lms_path_ex = self.lms_path_list[ex_int]
        
        img_real_T = self.process_img(img, lms_path, img_ex, lms_path_ex)
        audio_feat = self.get_audio_features(self.audio_feats, idx) # 
        # print(audio_feat.shape)
        if self.mode=="wenet":
            audio_feat = audio_feat.reshape(256,16,32)
        if self.mode=="hubert":
            audio_feat = audio_feat.reshape(32,32,32)
        y = torch.ones(1).float()
        
        return img_real_T, audio_feat, y

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class SyncNet_color(nn.Module):
    def __init__(self, mode):
        super(SyncNet_color, self).__init__()
        # 修改7.添加量化和反量化模块
        # self.quant = QuantStub()
        # self.dequant = DeQuantStub()

        self.face_encoder = nn.Sequential(
            Conv2d(3, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)
        
        p1 = 256
        p2 = (1, 2)
        if mode == "hubert":
            p1 = 32
            p2 = (2, 2)
        
        self.audio_encoder = nn.Sequential(
            Conv2d(p1, 256, kernel_size=3, stride=1, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(256, 256, kernel_size=3, stride=p2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 256, kernel_size=3, stride=2, padding=2),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, face_sequences, audio_sequences): # audio_sequences := (B, dim, T)
        #修改8.对输入进行量化
        # face_sequences = self.quant(face_sequences)
        # audio_sequences = self.quant(audio_sequences)

        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        #修改9.对输出进行反量化
        # audio_embedding = self.dequant(audio_embedding)
        # face_embedding = self.dequant(face_embedding)

        return audio_embedding, face_embedding

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

# def prepare_qat(model):
#     model.train()
#     # model.fuse_model()  # 可选：如果模型中有Conv+BN+ReLU结构，可融合提高效率
#     model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
#     torch.quantization.prepare_qat(model, inplace=True)
#     return model

def train(save_dir, dataset_dir, mode, num_epochs, batch_size):# 修改1.新增参数num_epochs和batch_size
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    train_dataset = Dataset(dataset_dir, mode=mode)
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,# 修改5，修改batch_size为传入的参数
        num_workers=4)
    model = SyncNet_color(mode).cuda()
    # model = prepare_qat(model)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=0.001)
    for epoch in range(num_epochs):
        # model.train()# 修改2.修改40为num_epochs
        for batch in train_data_loader:
            imgT, audioT, y = batch
            imgT = imgT.cuda()
            audioT = audioT.cuda()
            y = y.cuda()
            audio_embedding, face_embedding = model(imgT, audioT)
            loss = cosine_loss(audio_embedding, face_embedding, y)
            loss.backward()
            optimizer.step()
        print(epoch, loss.item())
        filename = f"epoch_{epoch}_loss_{loss.item():.7f}.pth" # 修改4.修改loss保存格式
        torch.save(model.state_dict(), os.path.join(save_dir, filename))
        # 保存中间量化模型（未转换成int8）
        # torch.save(model.state_dict(), os.path.join(save_dir, f"qat_epoch_{epoch}.pth"))

    # model.eval()
    # quantized_model = torch.quantization.convert(model.eval().cpu(), inplace=False)
    # torch.save(quantized_model.state_dict(), "quantized_model.pth")

    # 对比评估推理性能
    # evaluate_inference_speed_and_accuracy(model.cuda(), quantized_model, dataset_dir, mode)
def evaluate_inference_speed_and_accuracy(model_fp32, model_int8, dataset_dir, mode, num_samples=100):
    print("\n[INFO] Evaluating inference speed and cosine similarity (accuracy)...")
    test_dataset = Dataset(dataset_dir, mode=mode)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model_fp32.eval()
    model_int8.eval()

    cosine = nn.CosineSimilarity(dim=1)
    total_time_fp32 = 0.0
    total_time_int8 = 0.0
    sim_fp32 = 0.0
    sim_int8 = 0.0

    with torch.no_grad():
        for idx, (imgT, audioT, _) in enumerate(test_loader):
            if idx >= num_samples:
                break
            imgT = imgT.cpu()
            audioT = audioT.cpu()

            # 测量 FP32
            start = time.time()
            a_fp32, v_fp32 = model_fp32(imgT, audioT)
            total_time_fp32 += time.time() - start
            sim_fp32 += cosine(a_fp32, v_fp32).item()

            # 测量 INT8
            start = time.time()
            a_int8, v_int8 = model_int8(imgT, audioT)
            total_time_int8 += time.time() - start
            sim_int8 += cosine(a_int8, v_int8).item()

    print(f"[FP32] Avg Inference Time: {total_time_fp32 / num_samples:.6f} s")
    print(f"[INT8] Avg Inference Time: {total_time_int8 / num_samples:.6f} s")
    print(f"[FP32] Avg Cosine Similarity: {sim_fp32 / num_samples:.6f}")
    print(f"[INT8] Avg Cosine Similarity: {sim_int8 / num_samples:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--asr', type=str)
    parser.add_argument('--epochs', type=int, default=40)  # 修改3.参数解析器添加epochs参数
    parser.add_argument('--batch_size', type=int, default=1)  # 修改4.参数解析器添加batch_size参数
    opt = parser.parse_args()
    
    # syncnet = SyncNet_color(mode=opt.asr)
    # img = torch.zeros([1,3,160,160])
    # # audio = torch.zeros([1,128,16,32])
    # audio = torch.zeros([1,16,32,32])
    # audio_embedding, face_embedding = syncnet(img, audio)
    # print(audio_embedding.shape, face_embedding.shape)
    train(opt.save_dir, opt.dataset_dir, opt.asr,opt.epochs, opt.batch_size) #修改6.train函数新增参数
