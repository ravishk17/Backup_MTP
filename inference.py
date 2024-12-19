from TSRM import *
from TEP import *
from topic_predictor import *
from caption_generator import *
from sklearn.decomposition import PCA
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
import numpy as np
# Inference Function
# def generate_dense_captions(video_features, feature_extractor_path, tsrm_path, topic_predictor_path, caption_generator_path, vocab_size):
#     # Load models
#     feature_extractor = TemporalEventProposal(input_dim=500, hidden_dim=512, num_proposals=10)
#     feature_extractor.load_state_dict(torch.load(feature_extractor_path))
#     feature_extractor.eval()

#     tsrm = TemporalSemanticRelationModule(input_dim=512, hidden_dim=512)
#     tsrm.load_state_dict(torch.load(tsrm_path))
#     tsrm.eval()

#     topic_predictor = TemporalEventProposal(input_dim=512, hidden_dim=512, num_proposals=10)
#     topic_predictor.load_state_dict(torch.load(topic_predictor_path))
#     topic_predictor.eval()

#     caption_generator = CaptionGenerator(vocab_size=vocab_size, embed_dim=300, hidden_dim=512)
#     caption_generator.load_state_dict(torch.load(caption_generator_path))
#     caption_generator.eval()

#     # Preprocess video features
#     pca = PCA(n_components=500)
#     processed_features = torch.tensor(pca.fit_transform(video_features), dtype=torch.float32).unsqueeze(0)

#     with torch.no_grad():
#         # Extract Proposals
#         proposals = feature_extractor(processed_features)

#         # Compute Temporal-Semantic Relations
#         event_relations = tsrm(proposals)

#         # Generate Captions
#         captions = []
#         for event_idx in range(event_relations.size(1)):
#             event_feature = event_relations[0, event_idx].unsqueeze(0)
#             caption = []
#             word_input = torch.tensor([0], dtype=torch.long).unsqueeze(0)  # Start token

#             for _ in range(20):  # Maximum caption length
#                 outputs = caption_generator(word_input, event_feature)
#                 word = outputs.argmax(dim=-1).item()
#                 if word == 1:  # End token
#                     break
#                 caption.append(word)
#                 word_input = torch.tensor([[word]], dtype=torch.long)

#             captions.append(caption)

#     return captions

# Preprocess video features for PCA
def preprocess_features(video_features, target_dim=500):
    """Flatten 3D video features, apply PCA, and reshape back."""
    # Ensure video_features is a tensor
    batch_size, num_frames, feature_dim = video_features.shape
    flattened_features = video_features.reshape(-1, feature_dim)  # Flatten to 2D array

    # Dynamically adjust n_components for PCA
    n_components = min(target_dim, flattened_features.shape[0], feature_dim)

    # Apply PCA
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(flattened_features)

    # Reshape back to batch structure
    reshaped_features = reduced_features.reshape(batch_size, num_frames, n_components)

    # Pad to target_dim if necessary
    if n_components < target_dim:
        padded_features = torch.zeros((batch_size, num_frames, target_dim), dtype=torch.float32)
        padded_features[:, :, :n_components] = torch.tensor(reshaped_features, dtype=torch.float32)
        return padded_features

    return torch.tensor(reshaped_features, dtype=torch.float32)


def preprocess_video_features_for_pca(video_features, target_dim=500):
    """
    Preprocess video features to match the expected input size of the GRU.
    Ensures PCA reduces the dimensionality to `target_dim`.
    """
    # Find the maximum number of frames
    max_frames = max(video.shape[0] for video in video_features)

    # Pad all video features to have the same number of frames
    padded_features = []
    for video in video_features:
        padded = np.zeros((max_frames, video.shape[1]), dtype=np.float32)
        padded[:video.shape[0], :] = video
        padded_features.append(padded)

    # Convert to a single 2D array for PCA
    flattened_features = np.vstack(padded_features)

    # Ensure PCA does not attempt more components than allowed
    n_samples, n_features = flattened_features.shape
    valid_n_components = min(n_samples, n_features, target_dim)

    # Apply PCA
    pca = PCA(n_components=valid_n_components)
    reduced_features = pca.fit_transform(flattened_features)

    # Reshape back to per-video format
    start_idx = 0
    reduced_features_per_video = []
    for video in padded_features:
        num_frames = video.shape[0]
        reduced_features_per_video.append(reduced_features[start_idx:start_idx + num_frames])
        start_idx += num_frames

    # Convert to tensors with the correct shape
    reduced_features_per_video = [torch.tensor(f, dtype=torch.float32) for f in reduced_features_per_video]
    return reduced_features_per_video


# def generate_dense_captions(video_features, feature_extractor_path, tsrm_path, topic_predictor_path, caption_generator_path, vocab_size):
#     # Load models
#     feature_extractor = TemporalEventProposal(input_dim=500, hidden_dim=512, num_proposals=10)
#     feature_extractor.load_state_dict(torch.load(feature_extractor_path))
#     feature_extractor.eval()

#     tsrm = TemporalSemanticRelationModule(input_dim=512, hidden_dim=512)
#     tsrm.load_state_dict(torch.load(tsrm_path))
#     tsrm.eval()

#     topic_predictor = TopicPredictor(input_dim=512, hidden_dim=256, num_topics=10)
#     topic_predictor.load_state_dict(torch.load(topic_predictor_path))
#     topic_predictor.eval()

#     caption_generator = CaptionGenerator(vocab_size=vocab_size, embed_dim=768, hidden_dim=512)
#     caption_generator.load_state_dict(torch.load(caption_generator_path))
#     caption_generator.eval()

#     print(f"Type of video_features: {type(video_features)}")
#     print(f"First element type: {type(video_features[0]) if isinstance(video_features, list) else 'N/A'}")
#     if isinstance(video_features, list):
#         print(f"Length of video_features: {len(video_features)}")
#         print(f"Shape of first element: {video_features[0].shape if hasattr(video_features[0], 'shape') else 'N/A'}")

#     # Ensure video_features is a tensor
#     # Stack the list of tensors into a single tensor
#     video_features = torch.stack(video_features, dim=0)

#     # video_features = torch.tensor(video_features, dtype=torch.float32)

#     # Preprocess video features
#     processed_features = preprocess_features(video_features, target_dim=500)

#     with torch.no_grad():
#         # Extract Proposals
#         proposals = feature_extractor(processed_features)

#         # Compute Temporal-Semantic Relations
#         event_relations = tsrm(proposals)

#         # Predict Topics
#         topic_scores = topic_predictor(event_relations.mean(dim=1))

#         # Generate Captions
#         captions = []
#         for event_idx in range(event_relations.size(1)):
#             event_feature = event_relations[0, event_idx].unsqueeze(0)
#             caption = []
#             word_input = torch.tensor([0], dtype=torch.long).unsqueeze(0)  # Start token

#             for _ in range(20):  # Maximum caption length
#                 outputs = caption_generator(word_input, event_feature)
#                 word = outputs.argmax(dim=-1).item()
#                 if word == 1:  # End token
#                     break
#                 caption.append(word)
#                 word_input = torch.tensor([[word]], dtype=torch.long)

#             captions.append(caption)

#     return captions, topic_scores

# def generate_dense_captions(video_features, feature_extractor_path, tsrm_path, topic_predictor_path, caption_generator_path,tokenizer_path, vocab_size):
#     
#     # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#     tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
#     # Load models
#     feature_extractor = TemporalEventProposal(input_dim=500, hidden_dim=512, num_proposals=10)
#     feature_extractor.load_state_dict(torch.load(feature_extractor_path))
#     feature_extractor.eval()

#     tsrm = TemporalSemanticRelationModule(input_dim=512, hidden_dim=512)
#     tsrm.load_state_dict(torch.load(tsrm_path))
#     tsrm.eval()

#     topic_predictor = TopicPredictor(input_dim=512, hidden_dim=256, num_topics=10)
#     topic_predictor.load_state_dict(torch.load(topic_predictor_path))
#     topic_predictor.eval()

#     caption_generator = CaptionGenerator(vocab_size=vocab_size, embed_dim=768, hidden_dim=512)
#     caption_generator.load_state_dict(torch.load(caption_generator_path))
#     caption_generator.eval()

    

#     # Preprocess video features
#     # video_features = torch.stack(video_features, dim=0)
#     video_features = pad_sequence(video_features, batch_first=True)
#     processed_features = preprocess_features(video_features, target_dim=500)

#     captions = []
#     topic_scores = []

#     with torch.no_grad():
#         # Extract Proposals
#         proposals = feature_extractor(processed_features)

#         # Compute Temporal-Semantic Relations
#         event_relations = tsrm(proposals)

#         # Predict Topics
#         topic_scores = topic_predictor(event_relations.mean(dim=1))

#         # Generate Captions
#         for event_idx in range(event_relations.size(1)):
#             event_feature = event_relations[0, event_idx].unsqueeze(0)
#             caption = []
#             word_input = torch.tensor([tokenizer.cls_token_id], dtype=torch.long).unsqueeze(0)  # Start token

#             for _ in range(20):  # Maximum caption length
#                 outputs = caption_generator(word_input, event_feature)
#                 word_id = outputs.argmax(dim=-1).item()
#                 if word_id == tokenizer.sep_token_id:  # End token
#                     break
#                 caption.append(word_id)
#                 word_input = torch.tensor([[word_id]], dtype=torch.long)

#             decoded_caption = tokenizer.decode(caption, skip_special_tokens=True)
#             captions.append(decoded_caption)

#     return captions, topic_scores

def generate_dense_captions(
    video_features,
    feature_extractor_path,
    tsrm_path,
    topic_predictor_path,
    caption_generator_path,
    tokenizer_path,
    vocab_size
):
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    processed_features = preprocess_video_features_for_pca(video_features, target_dim=500)
    batch_features = torch.stack(processed_features)
    # Load models
    feature_extractor = TemporalEventProposal(input_dim=500, hidden_dim=512, num_proposals=10)
    feature_extractor.load_state_dict(torch.load(feature_extractor_path))
    feature_extractor.eval()

    tsrm = TemporalSemanticRelationModule(input_dim=512, hidden_dim=512)
    tsrm.load_state_dict(torch.load(tsrm_path))
    tsrm.eval()

    topic_predictor = TopicPredictor(input_dim=512, hidden_dim=256, num_topics=10)
    topic_predictor.load_state_dict(torch.load(topic_predictor_path))
    topic_predictor.eval()

    caption_generator = CaptionGenerator(vocab_size=vocab_size, embed_dim=768, hidden_dim=512)
    caption_generator.load_state_dict(torch.load(caption_generator_path))
    caption_generator.eval()
    
    # Preprocess video features
    pca = PCA(n_components=500)
    # processed_features = torch.tensor(pca.fit_transform(video_features), dtype=torch.float32).unsqueeze(0)

    captions = []
    with torch.no_grad():
        proposals = feature_extractor(batch_features)
        event_relations = tsrm(proposals)
        topic_scores = topic_predictor(event_relations.mean(dim=1))
        for event_idx in range(event_relations.size(1)):
            event_feature = event_relations[0, event_idx].unsqueeze(0)
            caption = []
            word_input = torch.tensor([tokenizer.cls_token_id], dtype=torch.long).unsqueeze(0)

            for _ in range(20):  # Maximum caption length
                outputs = caption_generator(word_input, event_feature)
                word_id = outputs.argmax(dim=-1).item()

                # Debugging: Print intermediate outputs
                # print(f"Event feature: {event_feature}")
                # print(f"Model outputs: {outputs}")
                print(f"Generated token ID: {word_id}, Decoded token: {tokenizer.decode([word_id])}")

                if word_id == tokenizer.sep_token_id:  # End with SEP token
                    break

                caption.append(word_id)
                word_input = torch.tensor([[word_id]], dtype=torch.long)

            decoded_caption = tokenizer.decode(caption, skip_special_tokens=True)
            captions.append(decoded_caption)

    return captions, topic_scores


