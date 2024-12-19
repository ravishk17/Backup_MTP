import torch
# Custom Collate Function
def custom_collate(batch):
    video_features, annotations = zip(*batch)
    max_frames = max(f.shape[0] for f in video_features)

    # Ensure all features have the correct target dimension
    feature_dim = video_features[0].shape[1]
    if feature_dim != 500:
        raise ValueError(f"Feature dimensions must be 500, but got {feature_dim}")

    padded_features = torch.zeros((len(video_features), max_frames, feature_dim))
    for i, f in enumerate(video_features):
        padded_features[i, :f.shape[0], :] = f

    return padded_features, annotations

