import torch
import torch.nn as nn
import torch.nn.functional as F
from vit import ViTEmbedder
from hubert import AudioEmbedder
import warnings
warnings.filterwarnings("ignore")
import time
class AudioVisualModel(nn.Module):
    def __init__(self, temperature=2):
        """
        A simplified Audio-Visual model with two small regularization terms:
          - non-negative pressure
          - temperature >= 1 pressure

        Args:
            temperature (float): Softmax temperature for contrastive scaling
        """
        super().__init__()
        
        self.visual_embedder = ViTEmbedder()
        self.audio_embedder = AudioEmbedder()
        
        # Keep temperature as a learnable parameter
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))
        
        # Unfreeze the HuBERT model (optional, if you want to train end-to-end)
        for param in self.audio_embedder.hubert.parameters():
            param.requires_grad = True
        for param in self.audio_embedder.projection.parameters():
            param.requires_grad = True

    def compute_temporal_similarity_matrix(self, audio_feats, visual_feats):
        """
        Compute pairwise cosine similarities between audio tokens and visual tokens across time.
        
        Args:
            audio_feats:  (B, Na, D)
            visual_feats: (B, T, Nv, D)
        
        Returns:
            similarities: (B, Na, T, Nv)
        """
        # Expand a time dimension in audio to broadcast
        audio_feats = audio_feats.unsqueeze(2)  # (B, Na, 1, D)
        
        # Normalize both
        audio_feats = F.normalize(audio_feats, dim=-1)
        visual_feats = F.normalize(visual_feats, dim=-1)
        
        # Compute pairwise similarities
        similarities = torch.einsum('bamd,btvd->batv', audio_feats, visual_feats)
        
        # Scale by temperature
        similarities = similarities / self.temperature
        return similarities

    def aggregate_temporal_similarities(self, similarities):
        """
        Simplified aggregation:
          1) Max over the visual patches dimension => shape (B, Na, T)
          2) Mean over time dimension => shape (B, Na)
          3) Mean over audio tokens => shape (B)

        For the (B, B, Na, T, Nv) case, the same steps but keep track of pairs:
          1) Max => (B, B, Na, T)
          2) Mean => (B, B, Na)
          3) Mean => (B, B)

        Args:
            similarities: (B, Na, T, Nv) or (B, B, Na, T, Nv)

        Returns:
            clip_similarities: (B) or (B, B)
        """
        # 1) Max over the visual tokens dimension
        max_across_patches = similarities.max(dim=-1)[0]  # => (B, Na, T) or (B, B, Na, T)

        # 2) Mean across time dimension
        mean_across_time = max_across_patches.mean(dim=-1)  # => (B, Na) or (B, B, Na)

        # 3) Mean across audio tokens dimension
        clip_similarities = mean_across_time.mean(dim=-1)   # => (B) or (B, B)

        return clip_similarities

    def compute_all_similarities(self, audio_feats, visual_feats):
        """
        Compute pairwise similarities for all (B x B) pairs in the batch.

        Args:
            audio_feats:  (B, Na, D)
            visual_feats: (B, T, Nv, D)
        
        Returns:
            clip_similarities: (B, B)
            similarities:      (B, B, Na, T, Nv)
        """
        B = audio_feats.size(0)

        # Expand for cross-pair matching
        audio_feats = audio_feats.unsqueeze(1)     # => (B, 1, Na, D) => (B, B, Na, D)
        visual_feats = visual_feats.unsqueeze(0)   # => (1, B, T, Nv, D) => (B, B, T, Nv, D)

        # Normalize first
        normed_audio_feats = F.normalize(audio_feats, dim=-1)
        normed_visual_feats = F.normalize(visual_feats, dim=-1)

        # Compute raw similarities for all pairs => shape: (B, B, Na, T, Nv)
        similarities = torch.einsum('xyad,xytvd->xyatv', normed_audio_feats, normed_visual_feats)

        # Scale by temperature
        similarities = similarities / self.temperature

        # Now aggregate for clip-level similarities
        clip_similarities = self.aggregate_temporal_similarities(similarities)

        return clip_similarities, similarities

    def compute_contrastive_loss(self, clip_similarities):
        """
        Standard InfoNCE-like loss on the clip-level similarities (B, B).

        Args:
            clip_similarities: (B, B)
        Returns:
            contrastive_loss: scalar
        """
        B = clip_similarities.size(0)
        labels = torch.arange(B, dtype=torch.long, device=clip_similarities.device)

        # Audio-to-Visual direction
        log_prob_a2v = F.log_softmax(clip_similarities, dim=1)  # (B, B)
        losses_a2v = -log_prob_a2v[torch.arange(B), labels]

        # Visual-to-Audio direction
        log_prob_v2a = F.log_softmax(clip_similarities.t(), dim=1)  # (B, B)
        losses_v2a = -log_prob_v2a[torch.arange(B), labels]

        # Average the two directions
        contrastive_loss = 0.5 * (losses_a2v + losses_v2a).mean()
        return contrastive_loss

    def compute_regularization_losses(self, similarities):
        """
        Regularize:
          1) Non-negative pressure: penalize negative values in raw similarities
          2) Temperature >= 1: penalize if self.temperature < 1

        Args:
            similarities: (B, B, Na, T, Nv) matrix of raw similarities.

        Returns:
            reg_loss: scalar
        """
        # (1) Non-negative pressure
        #     We clamp negative portion and penalize with squared error
        negative_part = torch.clamp(similarities, max=0.0)  # negative or zero
        l_nonneg = (negative_part ** 2).mean()

        # (2) Temperature >= 1 pressure
        #     We encourage temperature to be at least 1
        temp_low = torch.clamp(torch.log(torch.tensor(1.0, device=similarities.device)) - torch.log(self.temperature), min=0) ** 4
        temp_high = torch.clamp(torch.log(self.temperature) - torch.log(torch.tensor(3.0, device=similarities.device)), min=0) ** 4
        l_temp = temp_low + temp_high 

        reg_loss = l_nonneg + l_temp
        return reg_loss

    def forward(self, frames, audio):
        """
        Forward pass:
          1) Embed
          2) Compute (B, B) similarities + raw 5D similarities
          3) InfoNCE loss + small regularization

        Args:
            frames: (B, T, C, H, W)
            audio:  (B, samples)
        """
        # 1) Embed frames & audio
        #start_time = time.time()
        visual_feats = self.visual_embedder(frames)  # => (B, T, Nv, D)
        audio_feats = self.audio_embedder(audio)     # => (B, Na, D)
        #end_time = time.time()
        #print(f"Time taken for embedding: {end_time - start_time} seconds")

        if self.training:
            # 2) All-pairs
            #start_time = time.time()
            clip_sims, similarities_5d = self.compute_all_similarities(audio_feats, visual_feats)
            # 3) Compute InfoNCE + Regularization
            contrastive_loss = self.compute_contrastive_loss(clip_sims)
            reg_loss = self.compute_regularization_losses(similarities_5d)
            #end_time = time.time()
            #print(f"Time taken for all-pairs: {end_time - start_time} seconds")

            total_loss = contrastive_loss + 0.3 * reg_loss
            return total_loss, contrastive_loss, reg_loss

        else:
            # Inference => Just compute per-sample temporal matrix
            similarities = self.compute_temporal_similarity_matrix(audio_feats, visual_feats)
            return similarities


if __name__ == "__main__":
    # Quick sanity test
    model = AudioVisualModel(temperature=2.5)
    model.train()

    batch_size = 2
    num_frames = 8

    frames = torch.randn(batch_size, num_frames, 3, 224, 224)
    audio = torch.randn(batch_size, 16331)

    # Training forward pass
    try:
        loss = model(frames, audio)
        print("Training forward pass success!")
        print("Loss:", loss.item())
    except Exception as e:
        print("Error during training:", str(e))
        raise e

    # Inference test
    try:
        model.eval()
        with torch.no_grad():
            similarities = model(frames, audio)
            print("Inference pass success!")
            print("Similarities shape:", similarities.shape)
            print("Min:", similarities.min().item(), "Max:", similarities.max().item())
    except Exception as e:
        print("Error during inference:", str(e))
        raise e
