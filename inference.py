import sys
from audio2bs import Audio2BS
from data_capitulator import get_dataloaders, probe_audio_length
import torch
import argparse
import os
import shutil
from tqdm import tqdm
import numpy as np
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import soundfile as sf
import json
import copy
from utils.add_face_expression import  add_eye_blink_to_wts
import glob
import pickle
from data_capitulator import Dataset

DEVICE = 'cuda'
MODEL_ID = "wbbbbb/wav2vec2-large-chinese-zh-cn"
JSON_TEMPLATE = "./assets/0000_bsweight_template_a2f2023_1_1.json"


@torch.no_grad()
def inference_from_audio_path(fps, model_path, audio_file_list):
    model_stem = "{}_{}".format(Path(model_path).parent.stem, Path(model_path).stem)
    state_dict = torch.load(model_path)
    model_params = state_dict['model']
    ppe = state_dict['ppe']
    bs_number = state_dict['bs_number']
    audio_root = Path(audio_file_list[0]).parent.stem
    data_id = "{}_{}".format(model_stem, audio_root)
    model = Audio2BS(ppe, bs_number)
    model.cuda()
    model.load_state_dict(model_params, bs_number)
    model.eval() 
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    results = {}
    for audio_file in audio_file_list:
        audio_input, sample_rate = sf.read(audio_file)
        input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values
        audio_length = probe_audio_length(audio_file)
        bswts_frm_num = int(audio_length*fps)
        input_values = input_values[None,:,:]
        input_values = input_values.to(DEVICE)
        predicted_bswts = model(input_values, bswts_frm_num)
        results[audio_file] = predicted_bswts

    return results, data_id


@torch.no_grad()
def inference_from_dataset(model_path, dataset_path, dataset_type):
    f = open(dataset_path,'rb')
    dataset = pickle.load(f)
    data_loaders = get_dataloaders(dataset)
    data_loader = data_loaders[dataset_type]
    model_stem = "{}_{}".format(Path(model_path).parent.stem, Path(model_path).stem)
    data_id = "{}_dataset_type_{}".format(model_stem, dataset_type)
    state_dict = torch.load(model_path)
    model_params = state_dict['model']
    ppe = state_dict['ppe']
    bs_number = state_dict['bs_number']
    model = Audio2BS(ppe, bs_number)
    model.cuda()
    model.load_state_dict(model_params)
    model.eval() 
    pbar = tqdm(enumerate(data_loader),total=len(data_loader))
    results = {}
    for i, (audio, bswts, bswts_frm_num, key) in pbar:
        key = key[0]
        predicted_bswts = model(audio, bswts_frm_num)
        results[key] = predicted_bswts
    return results, data_id

def write_to_json(wts_dict, output_pth, data_id):
    os.makedirs(output_pth, exist_ok=True)
    template = open(JSON_TEMPLATE, 'rb')
    template_dict = json.load(template)
     
    for key in wts_dict:
        audio_path = Path(key)
        stem = "{}_{}_{}".format(data_id, audio_path.parent.stem, audio_path.stem) 
        outfile  = os.path.join(output_pth, "predicted_{}.json".format(stem))
        wts = wts_dict[key]
        json_dict = copy.deepcopy(template_dict)
        json_dict['numFrames'] = len(wts)
        shape = np.array(wts).shape
        json_dict['numFrames'] = shape[0]
        json_dict['weightMat'] = wts
        outfile_obj = open(outfile, 'w')
        json.dump(json_dict, outfile_obj, indent=4)
        outfile_obj.close()

def export_bswts(wts_dict):
    """export bswts for blender blendshape
    1. remove negative weights from model inference by clipping them between [0,1]
    2. add eye blink event
    Args:
        wts_dict(dict): {"audio_path": blendshpae weights array}

    """
    template = open(JSON_TEMPLATE, 'rb')
    template_dict = json.load(template)
    facename = template_dict['facsNames']
 
    export_wts_dict = copy.deepcopy(wts_dict)
    for key in export_wts_dict:
        wts = export_wts_dict[key]
        wts = wts.cpu().detach().numpy()
        wts = np.squeeze(wts)
        wts_clip = wts.clip(0,1)
        wts_w_eye = add_eye_blink_to_wts(wts_clip, facename)
        wts_w_eye = wts_w_eye.tolist()
        export_wts_dict[key] = wts_w_eye
    return export_wts_dict
   

def get_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Inference A2BS model: dataset/audio')
    parser_audio_path = subparsers.add_parser("audio", help='inference from root dir of audio files')
    parser_dataset = subparsers.add_parser("dataset", help='inference from dataset')
    
    parser_audio_path.add_argument("--fps", type=int, default=30, help='fps')
    parser_audio_path.add_argument("--model_path", type=str, default=None, help='model checkpoint pth file') 
    parser_audio_path.add_argument("--audio_file_root", type=str, default=None, help='audio_file_root')
    
    parser_dataset.add_argument("--model_path", type=str, default=None, help='model checkpoint pth file')
    parser_dataset.add_argument("--dataset_path", type=str, default=None, help='dataset_path')
    parser_dataset.add_argument("--dataset_type", type=str, default='valid', help='dataset type: valid/test/train')
    parser_audio_path.set_defaults(func=model_inference_from_audio_path) 
    parser_dataset.set_defaults(func=model_inference_from_dataset) 
    args = parser.parse_args()
    args.func(args)

def model_inference_from_dataset(args):
    model_path = args.model_path
    dataset_path = args.dataset_path
    dataset_type = args.dataset_type 
    results, data_id = inference_from_dataset(model_path, dataset_path, dataset_type)
    output_path = "inference_{}".format(data_id)
    export_wts_dict = export_bswts(results)
    write_to_json(export_wts_dict, output_path, data_id)

def model_inference_from_audio_path(args):
    fps = args.fps
    model_path = args.model_path
    audio_file_root = args.audio_file_root
    audio_file_list = glob.glob("{}/*wav".format(audio_file_root))
    results, data_id  = inference_from_audio_path(fps, model_path, audio_file_list)
    output_path = "inference_{}".format(data_id)
    export_wts_dict = export_bswts(results)
    write_to_json(export_wts_dict, output_path, data_id)
 

if __name__ == "__main__":
    get_args()
