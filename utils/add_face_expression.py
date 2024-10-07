import numpy as np
import scipy
from pathlib import Path
import json
from utils.analyze_bswts import read_bswts, detect_index 
import copy

EYEBLINK_FRAME_NUM = 5

def add_eye_close(N, std, intensity):
    """
    gaussian pulase with N points
    std: 1-3 (in frame interval)
    """ 
    pulse = scipy.signal.gaussian(N,std=std)*intensity
    return pulse

def sample_position(frame_num, event_interval = 120, interval_dev = 10):
    """sample eye blink position in timeline
    Args:
        frame_num(int): total frame numbers
        event_interval(int): time interval between eye blink events.(unit in frame)
                              since we are using fps = 30, frame_interval = 120 means 120/30 = 4 (secs)
        interval_dev(int): amount of deviation from the position decided by event interval. 
    Return: 
        simulated_pos_clip(list of int): postiion in timeline sampled according to interval
    """
    
    normal_pos = list(range(event_interval, frame_num, event_interval))
    pos_dev = np.random.randint(interval_dev, size=(len(normal_pos))) - interval_dev/2
    simulated_pos = normal_pos + pos_dev
    simulated_pos_clip = np.clip(simulated_pos, 0, frame_num-10).astype(int)
    return simulated_pos_clip

def sample_eye_close(frame_num):
    """sample an eye blink event with 7 time frames.
    Args: 
        frame_num(int): total frame number
    """
    position = sample_position(frame_num)
    eye_close_event = []
    for pos in position:
        indices = list(range(pos - EYEBLINK_FRAME_NUM//2, pos + EYEBLINK_FRAME_NUM//2+1))
        indices = [k for k in indices if k < frame_num]
        indices = [k for k in indices if k > -1 ]
        eye_close_event.append(indices)
    
    eye_close_full_array = np.zeros(frame_num)
    # sample eye close intensity between 0.7-1.0, but stays at 1.0 seems more realistic
    # eye_close_intensity = np.random.randint(3, size=(len(eye_close_event)))*0.1 + 0.7
    eye_close_intensity = np.ones(len(eye_close_event))
    
    for eye_close, intensity in zip(eye_close_event, eye_close_intensity):
        # sample std of gaussian signal between[1,2]
        std = np.random.randint(EYEBLINK_FRAME_NUM//2) + 1
        eye_close_full_array[eye_close] = add_eye_close(len(eye_close), std, intensity)
    
    return eye_close_full_array

def add_eye_blink_to_wts(wts, facename):
    """
    Args:
        wts(array): [T, 46] T: frame number
        facename(list): list of blendshape names
    Retrun:
        wts: blendshape weights array with eyeClose updated
    """
    eye_close_list = ['eyesCloseL','eyesCloseR']
    eye_idx = detect_index(facename, eye_close_list)
    frame_num = wts.shape[0]
    eye_close_array = sample_eye_close(frame_num)
    new_wts = copy.deepcopy(wts) 
    for idx in eye_idx:
        new_wts[:,idx] = eye_close_array
    return new_wts
    
             

def add_eye_blink_to_json(jsonfile, output_file):
    facename, wts = read_bswts(jsonfile)
    wts = add_eye_blink_to_wts(wts, facename)
    data = json.load(open(jsonfile,'rb'))
    data['weightMat'] = wts.tolist()
    outfile_obj = open(str(output_file), 'w')
    json.dump(data, outfile_obj, indent=4)
    outfile_obj.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='add eye blink to json')
    parser.add_argument("--input_json", type=str, help='input BS json file')
    parser.add_argument("--output_json", type=str, help='output BS json file')
    args = parser.parse_args()
    json_path = args.input_json
    output_json_path = args.output_json
    add_eye_blink_to_json(json_path, output_json_path)
 
