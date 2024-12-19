from torch.utils.data import Dataset
# 1. Data Preprocessing
class VideoDataset(Dataset):
    def __init__(self, video_features, annotations):
        self.video_features = video_features
        self.annotations = annotations

    def __len__(self):
        return len(self.video_features)

    def __getitem__(self, idx):
        video_feature = self.video_features[idx]
        annotation = self.annotations[idx]
        return video_feature, annotation