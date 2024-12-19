# 5. Temporal-Linguistic NMS
def temporal_linguistic_nms(proposals, captions, scores, iou_threshold=0.5, similarity_threshold=0.5):
    """Remove duplicate proposals and captions."""
    selected = []
    while proposals:
        best_idx = scores.argmax()
        selected.append((proposals[best_idx], captions[best_idx]))
        proposals.pop(best_idx)
        captions.pop(best_idx)
        scores.pop(best_idx)

        # Filter proposals based on thresholds (placeholder for actual logic)
        proposals = [p for i, p in enumerate(proposals) if i != best_idx]

    return selected