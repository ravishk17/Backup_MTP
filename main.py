import torch
from TSRM import *
from topic_predictor import *
# from preprocess_dataset import *
from video_dataset import *
from TEP import *
from caption_generator import *
from train_model import *
from inference import *
from get_video_features_and_annotations import *
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.decomposition import PCA
from util import *

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# def process_annotations(annotations):
#     processed_annotations = []
#     for ann in annotations:
#         captions = [tokenizer.encode(caption, add_special_tokens=True) for caption in ann["captions"]]
#         targets = [torch.tensor(caption) for caption in captions]
#         processed_annotations.append({"events": ann["events"], "targets": targets})
#     return processed_annotations



# def apply_pca_to_video_features(video_features, n_components=500):
#     processed_features = []
    
#     for video in video_features:
#         # Dynamically determine the number of components
#         effective_components = min(n_components, video.shape[0], video.shape[1])
#         pca = PCA(n_components=effective_components)
        
#         # Apply PCA on each video's features
#         video_pca = pca.fit_transform(video.numpy())
#         processed_features.append(torch.tensor(video_pca))
    
#     return processed_features

# Instantiate and Train
if __name__ == "__main__":
    # Placeholder: Load dataset and annotations
    video_features,annotations = get_video_features_and_annotations()  # Replace with actual feature loading logic # Replace with actual annotations loading logic
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = VideoDataset(video_features, annotations, target_dim=500)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate)

    # Initialize models
    vocab_size = tokenizer.vocab_size
    feature_extractor = TemporalEventProposal(input_dim=500, hidden_dim=512, num_proposals=10)
    tsrm = TemporalSemanticRelationModule(input_dim=512, hidden_dim=512)
    topic_predictor = TopicPredictor(input_dim=512, hidden_dim=256, num_topics=10)
    caption_generator = CaptionGenerator(vocab_size=vocab_size, embed_dim=768, hidden_dim=512)  # Using BERT vocab size

    # Train the models
    train_model(
        feature_extractor,
        tsrm,
        topic_predictor,
        caption_generator,
        dataloader,
        num_epochs=25,
        save_path="./models"
    )
