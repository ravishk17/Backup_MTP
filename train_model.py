import os
import torch
import torch.nn as nn
import torch.optim as optim
from save_and_load import *
from transformers import BertTokenizer
# 8. Training Loop
def train_model(feature_extractor, tsrm, topic_predictor, caption_generator, dataloader, num_epochs, save_path):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    optimizer = optim.Adam(list(feature_extractor.parameters()) +
                            list(tsrm.parameters()) +
                            list(topic_predictor.parameters()) +
                            list(caption_generator.parameters()), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    for epoch in range(num_epochs):
        for video_features, annotations in dataloader:
            optimizer.zero_grad()

            # Extract Proposals
            proposals = feature_extractor(video_features)

            # Compute Temporal-Semantic Relations
            event_relations = tsrm(proposals)

            # Predict Topics
            topic_scores = topic_predictor(event_relations.mean(dim=1))  # Mean pooling for simplicity

            # Generate Captions
            batch_loss = 0
            for idx, ann in enumerate(annotations):
                events = torch.tensor(ann["events"], dtype=torch.float32)  # Event timings
                captions = ann["captions"]  # Tokenized sentences corresponding to each event
                features = event_relations[idx]  # Features for the current video

                for event_idx, caption in enumerate(captions):
                    event_feature = features[event_idx].unsqueeze(0)  # Feature for the specific event
                    caption_tensor = torch.tensor(caption, dtype=torch.long).unsqueeze(0)
                    outputs = caption_generator(caption_tensor, event_feature)

                    targets = torch.tensor(caption, dtype=torch.long)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                    batch_loss += loss

            batch_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {batch_loss.item()}")

    # Save the final models
    torch.save(feature_extractor.state_dict(), os.path.join(save_path, "feature_extractor.pth"))
    torch.save(tsrm.state_dict(), os.path.join(save_path, "tsrm.pth"))
    torch.save(topic_predictor.state_dict(), os.path.join(save_path, "topic_predictor.pth"))
    torch.save(caption_generator.state_dict(), os.path.join(save_path, "caption_generator.pth"))
    tokenizer.save_pretrained(save_path)

