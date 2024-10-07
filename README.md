# Audio2blendshape
## create dataset_list

the dataset contains two parts, audio and blendshape weights
under the root of audio files folder it should contain subfolder for different speakers
```

├── speaker1
│   └── [xxxx%04d].wav
├── speaker2
│   └── [xxxx%04d].wav
└── speaker3
    └── [xxxx%04d].wav
Similarly, under root of blendshape weights json files, it contains subfolder for different speakers too

├── speaker1
│   └── [xxxx%04d].json
├── speaker2
│   └── [xxxx%04d].json
└── speaker3
    └── [xxxx%04d].json


python data_loader.py 
--audio_root /volume/smart-city-nas12/sfm/hsiaohsl/dataset/audio2bs/audio 
--bs_root /volume/smart-city-nas12/sfm/hsiaohsl/dataset/audio2bs/BS46/mfsi_400/mfsi_400 --no_eye
```
data list mfsi_400_audio_datalist_wo_eyes.p will be generated under bs_root folder

## train:
```
python train.py 
--save_path model_hhtest 
--logdir log_hhtest --data_list_path /volume/smart-city-nas12/sfm/hsiaohsl/dataset/audio2bs/BS46/mfsi_400/mfsi_400/mfsi_400_audio_datalist_wo_eyes.p 
--fps 30 --epoch 10

```
## inference:

### from dataset
```
python inference.py dataset --fps 30 
--model_path model_hhtest/10_model.pth 
--data_list_path /volume/smart-city-nas12/sfm/hsiaohsl/dataset/audio2bs/BS46/mfsi_400/mfsi_400/mfsi_400_audio_datalist_wo_eyes.p 
--dataset_type valid
```
a folder named "inference_10_model_dataset_type_valid" which contains bs weights json will be create where you run inference script
### from audio file root dir
```
python inference.py audio --fps 30 
--model_path model_hhtest/10_model.pth 
--audio_file_root /volume/smart-city-nas12/sfm/hsiaohsl/dataset/audio2bs/audio/ptt_sing_16000
```
a folder named "inference_10_model_ptt_sing_16000" which contains bs weights json will be create where you run inference script
