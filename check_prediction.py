import torch
from inference import *
from get_video_features_and_annotations import *
from transformers import BertTokenizer
# Example inference

# test_video_features = torch.randn(100, 4096)  # Raw features for a single video
test_video_features = get_only_video_features('data/features/c3d_features/c3d_features.hdf5')
# Generate captions
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer_path = "./models"
vocab_size = tokenizer.vocab_size
captions = generate_dense_captions(
    test_video_features,
    "./models/feature_extractor.pth",
    "./models/tsrm.pth",
    "./models/topic_predictor.pth",
    "./models/caption_generator.pth",
    tokenizer_path,
    vocab_size=vocab_size  # BERT vocab size
)

print("Generated Captions:", captions)