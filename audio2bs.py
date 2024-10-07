import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from wav2vec2ForA2BS import Wav2Vec2ForA2BS
import logging

MODEL_ID = "wbbbbb/wav2vec2-large-chinese-zh-cn"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PeriodicPositionalEncoding(nn.Module):
    """
    borrowed from FaceFormer
    "https://github.com/EvelynFan/FaceFormer/blob/main/faceformer.py"
    """
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=6000):
    #def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# linear interpolation layer
def linear_interpolation(features, frame_num, output_len=None):
    features = features.transpose(1, 2)
    output_features = F.interpolate(features,size=frame_num,align_corners=True,mode='linear')
    return output_features.transpose(1, 2)   

class Audio2BS(nn.Module):
    def __init__(self, use_ppe, bs_number):
        super(Audio2BS, self).__init__()
        """Audio2BS model 
        model sturcture please refers to
        "Learning Audio-Driven Viseme Dynamics for 3D Face Animation"
        """
        self.bs_number = bs_number
        self.ppe_feature_dim = 512
        self.ppe_period = 125
        self.use_ppe = use_ppe
        logger.info("use_ppe {}".format(self.use_ppe))
        self.wav2vec2 = Wav2Vec2ForA2BS.from_pretrained(MODEL_ID)
        self.wav2vec2._freeze_parameters()
        self.encode_linear = nn.Linear(1024,512)
        self.rnn  = nn.LSTM(512, 512, 1, batch_first = True, bidirectional = True)
        self.fully_connected = nn.Linear(512, self.bs_number)
        self.max_seq_len = 6000
        # periodic positional encoding 
        self.PPE = PeriodicPositionalEncoding(self.ppe_feature_dim, period = self.ppe_period, max_seq_len=self.max_seq_len)
        

    def decoder(self, features):
        """
        Input:
            features(torch.tensor): [1, T, 512] or [batch, T, 512] with batch = 1 here. T is frame number for 3D animation
        Output:
            final_bs_weights(torch.tensor): [1, T, 46] , G is frame number for 3D animation
        """
        outputs,_ = self.rnn(features)
        bi_direction_sum = outputs[:,:,:512] + outputs[:,:,512:]
        final_bs_weights = self.fully_connected(bi_direction_sum)
        return final_bs_weights

    def encoder(self, wav_input, frame_num):
        '''
        Input:
            wav_input(torch.tensor) [1, N], N is the lenght of enbemddings proportional to audio_lenght 
            frame_num(int): final frame number for 3D animation
        Return:
            vec2_interp(torch.tensor) [1, frame_num, 512] 
            
        '''
        vec2 = self.wav2vec2(torch.squeeze(wav_input,dim=0)).last_hidden_state
        vec2_512 = self.encode_linear(vec2)
        vec2_interp =  linear_interpolation(vec2_512, frame_num)
        vec2_w_ppe =  self.PPE(vec2_interp)
        if self.use_ppe:
            return vec2_w_ppe
        else:
            return vec2_interp


    def forward(self, wav_input, frame_num):
        z = self.encoder(wav_input, frame_num)
        x = self.decoder(z)
        return x

