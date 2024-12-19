import numpy as np
import h5py
import torch
import json

def get_video_features_and_annotations(file_path='data/features/c3d_features/c3d_features.hdf5', annotation_path = 'data/features/c3d_features/youcookii_annotations_trainval.json'):
    video_features = []
    annotations = []
    with open(annotation_path,'r') as annotations_file:
        total_annotations = json.load(annotations_file)
    total_annotations = total_annotations['database']
    
    with h5py.File(file_path, 'r') as f:
        for video_name in f.keys():
            group = f[video_name]
            curr_annotation = total_annotations[video_name]
            events = []
            captions = []
            for data in curr_annotation['annotations']:
                seg = data['segment']
                events.append(seg)
                sentence = data['sentence']
                captions.append(sentence)
            annotations.append({'events':events, 'captions':captions})

            features = np.array(group['c3d_features'][()])
            video_features.append(torch.from_numpy(features).float())

    return video_features,annotations

def get_only_video_features(file_path='data/features/c3d_features/c3d_infer_features.hdf5'):
    video_features = []
    with h5py.File(file_path, 'r') as f:
        for video_name in f.keys():
            group = f[video_name]
            features = np.array(group['c3d_features'][()])
            video_features.append(torch.from_numpy(features).float())

    return video_features