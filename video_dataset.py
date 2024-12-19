import torch
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from transformers import BertTokenizer

# Simplified Dataset Class
class VideoDataset(Dataset):
    def __init__(self, video_features, annotations, target_dim=500):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.annotations = self.tokenize_captions(annotations)
        self.video_features = self.preprocess_features(video_features, target_dim)

    # def preprocess_features(self, video_features, target_dim):
    #     """Reduce feature dimensions to target_dim using PCA."""
    #     pca = PCA(n_components=target_dim)
    #     processed_features = []
    #     for video in video_features:
    #         reduced_features = pca.fit_transform(video)
    #         processed_features.append(torch.tensor(reduced_features, dtype=torch.float32))
    #     return processed_features
    
    # def preprocess_features(self, video_features, target_dim):
    #     processed_features = []
    #     for video in video_features:
    #         pca = PCA(n_components=min(target_dim, video.shape[0], video.shape[1]))
    #         reduced_features = pca.fit_transform(video)
    #         processed_features.append(torch.tensor(reduced_features, dtype=torch.float32))
    #     return processed_features

    def preprocess_features(self, video_features, target_dim=500):
        processed_features = []
        for video in video_features:
            n_components = min(target_dim, video.shape[0], video.shape[1])  # Dynamically adjust n_components
            pca = PCA(n_components=n_components)
            reduced_features = pca.fit_transform(video)
            # If reduced_features has fewer dimensions than target_dim, pad with zeros
            padded_features = torch.zeros((reduced_features.shape[0], target_dim))
            padded_features[:, :reduced_features.shape[1]] = torch.tensor(reduced_features, dtype=torch.float32)
            processed_features.append(padded_features)
        return processed_features




    # def tokenize_captions(self, annotations):
    #     """Tokenize captions using BERT tokenizer."""
    #     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    #     for annotation in annotations:
    #         annotation["captions"] = [
    #             tokenizer.encode(caption, add_special_tokens=True) for caption in annotation["captions"]
    #         ]
    #     return annotations
    
    def tokenize_captions(self, annotations):
        """Tokenize captions using the provided tokenizer."""
        for annotation in annotations:
            annotation["captions"] = [
                self.tokenizer.encode(
                    caption,
                    add_special_tokens=True,
                    padding="max_length",
                    truncation=True,
                    max_length=30  # Maximum caption length
                ) for caption in annotation["captions"]
            ]
        return annotations

    def __len__(self):
        return len(self.video_features)

    def __getitem__(self, idx):
        return self.video_features[idx], self.annotations[idx]