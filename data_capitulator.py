import os
import torch
from collections import defaultdict
from torch.utils import data
import copy
import numpy as np
import pickle
import subprocess
from tqdm import tqdm
import random,math
from transformers import Wav2Vec2Processor
import soundfile as sf
import glob
from pathlib import Path
import json
import logging
import argparse
from torch.utils.data import random_split
from utils.analyze_bswts import generate_MSEweights, EYE_NAMES, eyes_idx


MODEL_ID = "wbbbbb/wav2vec2-large-chinese-zh-cn"
DEVICE = 'cuda'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data):
        self.data = data
        self.len = len(self.data)

    def __getitem__(self, index):
        """Returns one data pair (source and target).
           subject key will be reture as tuple with uknown reason
        Args:
            index (int): index to fetch data from dataset
        Returns:
            audio (torch.tensor): [1,1,N], processed audio signal by Wav2Vec2Processor. N will be realated to the audio length 
            bswts (torch.tensor): [1,M,46]. blendshape weights. M is number of key frames of this audio file, and 46 correponds to 46 shapekeys.
            bswts_frm_num (tor.tnesor): [a]  frame numbers. denoted as M.
            data_key(str): key of this data in dataset. Designed as {ID of source voice}_{serial num of data}. ex: {HH}_{0001}, 0001.wav spoken by HH.
                              WARNING!! data_key is in type str, but will return as tuple for unknow reason.
        """
        # seq_len, fea_dim
        audio = self.data[index]["audio"]
        bswts = self.data[index]["bswts"]
        bswts_frm_num = self.data[index]["bswts_frm_num"]
        data_key = self.data[index]["data_key"]
        return torch.FloatTensor(audio).to(DEVICE), torch.FloatTensor(bswts).to(DEVICE), bswts_frm_num, data_key

    def __len__(self):
        return self.len

def probe_audio_length(audio_path: str, timeout=None):
    """detect audio length in secs
    Input: 
        audio_path (str) : audio file path
        timeout (int): timeout in secs
    Return:
        audio_length (float): audio length in secs
        
    """
    ffprobe_cmd = ['ffprobe', 
                    '-i', audio_path, 
                    '-show_entries', 'format=duration', 
                    '-v', 'error', 
                    '-of', 'csv=p=0']
    try:
        output = subprocess.run(
            ffprobe_cmd,
            shell=False,
            check=True,
            timeout=timeout,
            capture_output=True)
        audio_length = output.stdout.decode()
    except subprocess.CalledProcessError as err:
        msg = (
            f'FFprobe command [{ffprobe_cmd}] failed.\n'
            f'Returncode: {err.returncode}\n'
            f'Error msg: {err.stderr.decode("utf-8")}')
        raise Exception(msg)
    return float(audio_length)

def load_bswts(bswts_json_file):
    """load blendshape json file
    Input:
        bswts_json_file(str): json file path
    Return:
        num_frames(int): number of frame numbers
        num_bs(int): number of blendshape keys (46/52)
        face_names(str): list of str, names of blenshape keys.
        bswts(float): [frame_nums, face_names] list of blendshape wieghtes
    """
    bswts_data = json.load(open(bswts_json_file))
    num_frames = bswts_data['numFrames']
    num_bs = bswts_data['numPoses']
    face_names = bswts_data['facsNames']
    bswts = torch.Tensor(bswts_data['weightMat'])
    return num_frames, num_bs, face_names, bswts

def zero_wts(bswts, idx_list):
    """set blendshape keys to zero corresponding to idx_list
    Input:
        bswts(list of float): [N, 46] N: number os frames
        idx_list(list of int): ex [1,3,4]
    Returns:
        bswts(list of float): [N, 46] with bs weights set to zero
    """
    bswts[:,idx_list] = 0.0
    return bswts

def gen_key(parent_path, full_path):
    """Generate key for individual data and subject ID
    for example:
    parent path /volume/annotator-share-nas12/HH/3Davatar/BSweights
    full_path = /volume/annotator-share-nas12/HH/3Davatar/BSweights/msi_sub/0095.json
    
    subject_id = 'msi_sub'
    data_key = "msi_sub_0095"
    
    Inputs:
        parent path(str): ex: /volume/annotator-share-nas12/HH/3Davatar/BSweights
        full_path(str): ex: /volume/annotator-share-nas12/HH/3Davatar/BSweights/msi_sub/0095.json
    Returns:
        data_key(str): key corresponding to particular data
        subject_id(str): the source of voice generateing this data(who speaks)
    """
    ppth = Path(parent_path)
    fpth = Path(full_path)
    relative_pth = fpth.relative_to(ppth)
    parts = relative_pth.parts
    parts_list = [p for p in parts]
    file_stem = relative_pth.stem
    file_stem_numerics = extract_numerics(file_stem)
    parts_list[-1] = file_stem_numerics
    data_key = "_".join(parts_list)
    subject_id = parts_list[0]
    return data_key, subject_id

def extract_numerics(str_in):
    """extract numetric part from a string
    Input:
        str_in(str): input string
    Return:
        num(str): numeric part of the input string
    """
    num = ""
    for c in str_in:
        if c.isdigit():
            num = num + c
    return num

def fill_bswts(data, bswts_json_path_root, bswts_path_list, no_eyes=True):
    """fill blendshape weights data into data dict
    Inputs:
        data(dict): deictionary contains dataset info
        bswts_json_path_root(str): directory contains GT blendshape json files
        bswts_path_list(list of str): list of file path of GT blendshape json files
        no_eyes(bool): True: set eye-related blendshapes weights to zero. False: leave them as original.
    Outpus:
        data(dict): deictionary contains dataset info
    """
    for bswts_path in bswts_path_list:
        num_frames, num_bs, face_names, bswts =  load_bswts(bswts_path)
        eye_idx_list = eyes_idx(face_names)
        if no_eyes:
            mouth_bswts = zero_wts(bswts, eye_idx_list)
        else:
            mouth_bswts = bswts
        key, subject_id = gen_key(bswts_json_path_root, bswts_path)
        data[key]['bswts'] = torch.Tensor(mouth_bswts)
        data[key]['bswts_frm_num'] = num_frames
        data[key]['subject_id'] = subject_id
        data[key]['data_key'] = key
    return data

def fill_audio(data, audio_path_root, audio_path_list):
    """fill audio data into data dict
    Inputs:
        data(dict): deictionary contains dataset info
        audio_path_root(str): directory contains input audio files
        audio_path_list(list of str): list of file path input audio files
    Outputs:
        data(dict): deictionary contains dataset info
    """ 
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID) 
    for audio_path in tqdm(audio_path_list):
        key, _  = gen_key(audio_path_root, audio_path)
        if key in data.keys():
            audio_input, sample_rate = sf.read(audio_path) 
            input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values 
            audio_length = probe_audio_length(audio_path)
            data[key]['audio'] = input_values
            data[key]['audio_len'] = float(audio_length)
    return data

def prepare_dataset(wav_path_root, bswts_path_root, save_path, no_eyes=True, fps=30, train_frac=0.96, test_frac=0.01, valid_frac=0.03, valid_mean_percent=90, top_list_percent=50):
    """generate dataset dict 
    dataset['train/valid/test'] contains Dataset object
    Dataset object contains:
        bswts: blend shape weights
        bswts_frm_num: frame number
        subject_id: audio suject ID (who's voice)
        data_key: key of this particular data in dataset
    
    dataset['mse_loss_weight'] contains weights for weightedMSELosee
    
    Inputs:
        wav_path_root(str): folder path contains audio wav files
        bswts_path_root(str): folder path contains blendshape json file
        save_path(str): folder path to save data list 
        no_eyes(bool): True: set eye-related blendshapes weights to zero. False: leave them as original.
        fps(int): fps
        train_frac(float): fraction of training set
        test_frac(float): fraction of test set
        valid_frac(float): fraction of validataion set
        valid_mean_percent(float): percentile of blendshape data that to be used to calculate means of blendshape weights
        top_list_percent(float): percentile of the mean BS wights that to be included in top list 
 
    """
    audio_path_root = wav_path_root
    bswts_json_path_root = bswts_path_root
  
    audio_path_list = glob.glob('{}/**/*wav'.format(audio_path_root),recursive=True) 
    bswts_path_list = glob.glob('{}/**/*json'.format(bswts_json_path_root),recursive=True) 
    data = defaultdict(dict)
    data = fill_bswts(data, bswts_json_path_root, bswts_path_list, no_eyes=no_eyes)
    data = fill_audio(data, audio_path_root, audio_path_list)
    
    data_list = []
    for key in data:
        v = data[key]
        data_list.append(v)    

    os.makedirs(save_path, exist_ok=True)  
    wav_name = os.path.basename(wav_path_root)
    bswts_name = os.path.basename(bswts_path_root)
    if no_eyes:
        eye_stem = 'wo_eyes'
    else:
        eye_stem = "w_eyes"
   
    if check_dataset(data_list, fps):
        logger.info('dataset OK')
   
    # generate MSE weights
    mse_loss_weight = generate_MSEweights(data_list, valid_mean_percent=valid_mean_percent, top_list_percent=top_list_percent)
    logger.info("mse_loss_weight {}".format(mse_loss_weight))
 
    train_idx, test_idx, val_idx = random_split(data_list, [train_frac, test_frac, valid_frac], generator=torch.Generator().manual_seed(42))
    
    train_list = [data_list[i] for i in train_idx.indices]
    test_list = [data_list[i] for i in test_idx.indices]
    valid_list = [data_list[i] for i in val_idx.indices]
    
    train_data = Dataset(train_list)
    test_data = Dataset(test_list)
    valid_data = Dataset(valid_list)
    
    dataset = {}
    dataset['train'] = train_data
    dataset['valid'] = valid_data
    dataset['test'] = test_data
    dataset['mse_loss_weight'] = mse_loss_weight 
    
    save_file = os.path.join(save_path, '{}_{}_dataset_{}.p'.format(bswts_name, wav_name, eye_stem))
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f)

def check_dataset(data_list, fps):
    """check if dataset has wrong audio-blendshape weights pair
    we only compare if the frame number is consistent
    Inputs:
        data_list: list of data dict
        fps(int): fps 
    """
    for data in data_list:
        diff = int(data['audio_len']*fps) - data['bswts_frm_num']
        if abs(diff) > 1:
            msg = 'key {} has inconsistency in auido frame and bs frame'.format(data['subject_key'])
            raise Exception(msg)
    return True

def get_dataloaders(dataset):
    """
    Inputs:
        dataset: dict contains Dataset objects
    Return:
        data_loader: dict contains data.DataLoader object
    """

    train_data = dataset['train']
    valid_data = dataset['valid']
    test_data = dataset['test']

    data_loader = {}
    data_loader["train"] = data.DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    data_loader["valid"] = data.DataLoader(dataset=valid_data, batch_size=1, shuffle=True)
    data_loader["test"] = data.DataLoader(dataset=test_data, batch_size=1, shuffle=True)
    return data_loader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/config_mfsi_400_dataset.json', help='config file')
    args = parser.parse_args()
    return args
    
def create_dataset():
    args = get_args()
    config_file = args.config
    config = json.load(open(config_file,'rb'))
    audio_root = config['audio_root']
    bs_root = config['bs_root']
    no_eye = config['no_eyes']
    if no_eye == 1:
        no_eyes = True
    else:
        no_eyes = False
    fps = int(config['exportFps'])
    train_frac = config['train_frac']
    test_frac = config['test_frac']
    valid_frac = config['valid_frac']
    valid_mean_percent = config['valid_mean_percent']
    top_list_percent = config['top_list_percent']
    prepare_dataset(audio_root, bs_root, bs_root, no_eyes=no_eyes, fps=30, train_frac=0.96, test_frac=0.01, valid_frac=0.03, valid_mean_percent=90, top_list_percent=50)

if __name__ == "__main__":
    create_dataset()

