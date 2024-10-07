import json
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import pickle 
import logging
import glob
import argparse
import os


JSON_TEMPLATE = "./assets/0000_bsweight_template_a2f2023_1_1.json"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EYE_NAMES = ['browLowerL',
            'browLowerR',
            'innerBrowRaiserL',
            'innerBrowRaiserR',
            'outerBrowRaiserL',
            'outerBrowRaiserR',
            'eyesLookLeft',
            'eyesLookRight',
            'eyesLookUp',
            'eyesLookDown',
            'eyesUpperLidRaiserL',
            'eyesUpperLidRaiserR',
            'squintL',
            'squintR',
            'eyesCloseL',
            'eyesCloseR']
 

def eyes_idx(face_names):
    """detect index list corresping to blendshapes realted to eyes
    Input:
        face_names(list of strings): name list of blendshapes
    output:
        eye_idx(list of int): index corresponding to eye blendshapes
    """
    eye_idx = []
    for idx, f_name in enumerate(face_names):
        if f_name in EYE_NAMES:
            eye_idx.append(idx)
    return eye_idx

def read_bswts(json_file):
    """
    Args:
        json_file(str): input blendshape json file
    Return:
        facename(list(str)): list of name of blendshape keys 
        wts(array): [T, 46], T: frame number
    """
    data = json.load(open(json_file,'rb'))
    wts = data['weightMat']
    wts = np.array(wts)
    facename = data['facsNames']
    return facename, wts

def detect_index(face_bs_names, face_bs_list):
    """detect index of elements in face_list appears in facename.
    Args:
        face_bs_name(list(str)): list of name of blendshape keys
        face_bs_list(list(str)): list of subset of blendshape names
    Return:
        idx_list(list(int)): list of index in facename of face_list elements. 
    """
    idx_list = []
    for face_bs in face_bs_list:
        if face_bs in face_bs_names:
            idx = face_bs_names.index(face_bs)
            idx_list.append(idx)
    return idx_list

def plot_wts(wts_list, plot_idx, face_bs_list, label_list):
    """plot blendshape weights in differenct subplots from different models.
    Args: 
        wts_list(list(array)): list of blendshape weights arrays from difference models
        plot_idx(list(list(ubt))): plotting index of blendshape weights arrays from difference models
        face_bs_list(list(array)): list of blendshape wieght from difference models
        label_list(list(str)): labels of difference models
    """
    fig = plt.figure(figsize=(10,6)) 
    num_splot = len(face_bs_list)
    ax_list = []
    plot_idx = np.array(plot_idx).astype(np.int8).transpose()
    for idxsp in range(num_splot):
        a = "{}1{}".format(num_splot, idxsp+1)
        ax_list.append(fig.add_subplot(int(a)))
   
    for ax, bsname, pidx in  zip(ax_list, face_bs_list, plot_idx):
        ax.set_xlabel(bsname)
        for idx, (wt, label) in enumerate(zip(wts_list, label_list)):
            ax.plot(wt[:,pidx[idx]],label=label)
            ax.legend()
            
    fig.text(0.5, 0.01, 'video frames', ha='center')
    fig.text(0.04, 0.5, 'blendshape weights', va='center', rotation='vertical')
    plt.savefig('wts.png')         



def make_plots():
    """tool to make blendshape analysis plots
    """
    parser = argparse.ArgumentParser(description='ploting bs weights')
    parser.add_argument("--json_file_gt", type=str, help='GT bs json file')
    parser.add_argument("--json_file", type=str, help='model bs json file')
    parser.add_argument("--plot_list", type=str, default=None, nargs='+', help='list of blendshape name to be plotted')
    args = parser.parse_args()
    json_file_gt = args.json_file_gt
    json_file = args.json_file
    plot_list = args.plot_list
    fname_gt, wts_gt = read_bswts(json_file_gt)
    fname_model, wts_model = read_bswts(json_file)
    idx_list_gt = detect_index(fname_gt, plot_list)
    if len(idx_list_gt) < 1:
        print('no plotting list')
        return 
    idx_list_model = detect_index(fname_model, plot_list)
    wt_list = []
    wt_list.append(wts_gt)
    wt_list.append(wts_model)
    plot_idx_list = []
    plot_idx_list.append(idx_list_gt)
    plot_idx_list.append(idx_list_model)
    plot_wts(wt_list, plot_idx_list, plot_list,['gt','model'])

def generate_MSEweights(datalist, valid_mean_percent=90, top_list_percent=50):
    """
    Args:
        datalist(list(dict)): list of dict that contains datasdt info
    Return:
        mse_wts_full(array(N)): weights for weightedMSELoss. N is number of blendshape
    """
    bswts_full = get_bswts_from_data_list(datalist)
    fname, _ = read_bswts(JSON_TEMPLATE)
    mse_wts_sub, index = analyze_bswts(bswts_full,fname, valid_mean_percent=90, top_list_percent=50)
    mse_wts_full = gen_full_MSEweights(mse_wts_sub, index)
    return mse_wts_full

def analyze():
    """Tool to analyze blendshapes weights value within a dataset
    Args:
        file_root(str): root of BS dataset. should contain subfolders for blendshape json files.
        top_lsit_p(int): 
        valid_mean_percent(int): percentile of blendshape data that to be used to calculate means of blendshape weights
        top_list_percent(int): percentile of the mean BS wights that to be included in top list 
    Output:
        One can see MSE weights printed on console. One can experiment on the values of MSE weights by changing
        parameters "valid_mean_percent" and "top_list_percent".
    """
    parser = argparse.ArgumentParser(description='analyze bs weights')
    parser.add_argument("--file_root", type=str, help='input BS json file')
    parser.add_argument("--top_list_p", type=int, default=50, help='top list percent')
    parser.add_argument("--valid_mean_p", type=int, default=90, help='valid mean percent')
    args = parser.parse_args()
    file_root = args.file_root
    top_list_percent = args.top_list_p
    valid_mean_percent = args.valid_mean_p
    root_path = file_root
    folders = []
    for child in os.listdir(root_path):
        if os.path.isdir(os.path.join(root_path, child)):
            folders.append(child)

    fname_list = []
    wts_list  = []
    root_path = Path(root_path)
    
    ct = 0
    for folder in folders:
        full_folder = root_path / folder
        json_name_list = glob.glob("{}/*json".format(full_folder))
        for json_name in json_name_list:
            fname, wts = read_bswts(json_name)
            if ct == 0:
                wts_full = wts
            else:
                wts_full = np.concatenate((wts_full, wts), axis=0)
            ct = ct +1
    mse_wts, index = analyze_bswts(wts_full,fname, valid_mean_percent=valid_mean_percent, top_list_percent=top_list_percent)
    print("mse_wts {}".format(mse_wts))
    print("index {}".format(index))

def get_bswts_from_data_list(data_list):
    """collect all blendshape frames from a datalist
    Args:
        list(dict): list constains dataset data
    Return:
        bswts_full(array): [T, 46] all frames in datalist(comes from all audio source inputs)
    """
    for idx, data in enumerate(data_list):
        bswts = data['bswts']
        bswts = np.array(bswts)
        if idx == 0:
            bswts_full = bswts
        else:
            bswts_full = np.concatenate((bswts_full, bswts), axis=0)
    return bswts_full


def analyze_bswts(wts, face_bs_name, valid_mean_percent=90, top_list_percent=50):
    """analyze the blendshape weights from whole dataset, and decide the wight we want to apply on weighted MSE
       1. We exclude eye blendshapes and blendshape that has zero weights.(invlid)
       2. We caluculated mean blendshape weights from  their top X% values.
            X% is decided by vlid_mean_percent.(90 means top 10% of ata)
       3. We select the face blendshapes with their mean blendshape weights that are within top Y% among other. 
            Y: top_list_percent(50 means top 50 % of data)
       4. Convert  mean BS weights to MSE weights.
       
       In this setting, our mfsi_400 datast will have MSE weights looks like :
       
       [1.         1.         1.         1.         1.         1.
        1.         1.         1.         1.         1.         1.
        1.         1.         1.         1.         1.         1.
        1.         1.         1.         1.         1.         1.
        1.         1.11765993 2.08289289 1.45533812 1.41451406 1.
        2.45420074 1.         2.25606918 1.         1.         1.56335473
        1.         1.         1.12007475 1.38419354 1.23125947 1.19113231
        1.         2.12866807 1.         1.37993765]
        
    Args:
        wts(array): [T, 46]
        face_bs_name(list(str)):  list of full blendshape names
        valid_mean_percent(float): percentile of blendshape data that to be used to calculate means of blendshape weights
        top_list_percent(float): percentile of the mean BS wights that to be included in top list 
    Return:
        mean_wts(array): mean BS wheights that are on top 50% of valid blendshapes
        index_top_in_bs(array): index of mean_wts in face_bs_name
    """
    bs_num = 46
    valid_idx_list = []
    mean_wts_list = []
    eye_index = eyes_idx(face_bs_name)
    for i in range(bs_num):
        if i not in eye_index:
            values = wts[np.abs(wts[:,i]) > 0.01,i]
            if len(values) > 0:
                valid_idx_list.append(i)
                ptile = np.percentile(values, valid_mean_percent)
                mean = np.mean(values[values > ptile])
                mean_wts_list.append(mean)
    mean_wts = np.array(mean_wts_list) 
    face_bs_name = np.array(face_bs_name)
    ptile = np.percentile(mean_wts, top_list_percent)
    mean_wts_top = mean_wts[mean_wts > ptile]
    index_top = np.where(mean_wts > ptile)
    index_top_in_bs = np.array(valid_idx_list)[index_top]
    mse_wts = gen_MSE_wts(mean_wts_top)       
    return mse_wts, index_top_in_bs

def gen_full_MSEweights(mse_wts_sub, index):
    """ assign mse_wts_sub values to final MSE weights, and assign others to 1.
    Args: 
        mse_wts_sub(array): subset of MSE weight array
        index(array): correspoding index of subset of MSE weight array
    Return:
       mse_weights(array): [46] MSE weights corresponding to each blendshape keys.
    """
    mse_weights = np.ones(46)
    mse_weights[index] = mse_wts_sub
    return mse_weights

def gen_MSE_wts(bs_mean_wts):
    """generate MSE weights
        1. normaize according to max in bs_mean_wts
        2. gen MSE wts ~ 1/BS wts
    Args: 
        bs_mean_wts(array): mean BS weights.
    Return:
        mse_wts(array): weights for weighted MSE loss
    """
    norm = np.max(bs_mean_wts)
    mean_wts_normalized = bs_mean_wts/norm
    mean_wts_inv = 1./mean_wts_normalized
    mse_wts = mean_wts_inv
    return mse_wts

if __name__ == "__main__":
    #analyze()
    make_plots()
