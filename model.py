import torch
import torch.nn as nn
import torch.nn.functional as F
from vit import ViTEmbedder
from hubert import AudioEmbedder
import warnings
warnings.filterwarnings("ignore")

class AudioVisualModel(nn.Module):
    def __init__(self, temperature=2.5, initial_threshold=-2.5, scale_factor=2.5, alpha=10.0):
        """
        Args:
            temperature: Initial temperature for contrastive scaling
            initial_threshold: Initialized threshold parameter (before sigmoid)
            scale_factor: Multiplier for gating
            alpha: Sharpness factor for soft gating
        """
        super().__init__()
        
        self.visual_embedder = ViTEmbedder()
        self.audio_embedder = AudioEmbedder()
        
        # Learnable temperature & threshold & scale_factor
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.threshold = nn.Parameter(torch.tensor(initial_threshold))
        self.scale_factor = nn.Parameter(torch.tensor(scale_factor))
        
        # alpha is not necessarily learnable, we can keep it fixed
        self.alpha = alpha
        
        # Unfreeze the HuBERT model
        for param in self.audio_embedder.hubert.parameters():
            param.requires_grad = True
        for param in self.audio_embedder.projection.parameters():
            param.requires_grad = True

    def compute_temporal_similarity_matrix(self, audio_feats, visual_feats):
        """
        Compute pairwise cosine similarities between audio tokens and visual tokens across time
        
        Args:
            audio_feats:  (B, Na, D)
            visual_feats: (B, T, Nv, D)
        
        Returns:
            similarities: (B, Na, T, Nv)
        """
        # Expand a time dimension in audio
        audio_feats = audio_feats.unsqueeze(2)  # (B, Na, 1, D)
        
        # Normalize
        audio_feats = F.normalize(audio_feats, dim=-1)
        visual_feats = F.normalize(visual_feats, dim=-1)
        
        # Compute similarities
        similarities = torch.einsum('bamd,btvd->batv', audio_feats, visual_feats)
        
        # Scale by temperature
        return similarities / self.temperature

    def aggregate_temporal_similarities(self, similarities):
        """
        similarities: (B, Na, T, Nv) or (B, B, Na, T, Nv)
        
        Returns:
            clip_similarities: (B) or (B, B)
            fraction_selected: float
        """
        # 1. threshold in [0,1]
        threshold_val = torch.sigmoid(self.threshold)  # shape []
        
        # 2. max over visual tokens
        max_visual_similarities = torch.max(similarities, dim=-1)[0]  # shape => (B, Na, T) or (B, B, Na, T)
        
        # 3. Compute gating
        #    raw_diff = max_visual_similarities - threshold_val
        #    gating = sigmoid(alpha * raw_diff)
        #    selection_strength = gating * scale_factor
        raw_diff = max_visual_similarities - threshold_val
        gating = torch.sigmoid(self.alpha * raw_diff)
        selection_strength = gating * self.scale_factor
        
        # 4. Weighted similarities
        masked_similarities = max_visual_similarities * selection_strength
        
        # 5. Weighted average
        weighted_sum = masked_similarities.sum(dim=-1)  # sum over T
        weights_sum = selection_strength.sum(dim=-1)    # sum over T
        token_similarities = weighted_sum / weights_sum.clamp(min=1e-6)
        
        # (Optional) If you want to ensure token_similarities <= 1
        # token_similarities = token_similarities.clamp(0, 1)
        
        # 6. Average over audio tokens
        clip_similarities = token_similarities.mean(dim=-1)  # => shape (B) or (B, B)
        
        # fraction_selected for monitoring
        # "selected" means gating > 0.5? or raw_diff > 0? Or maybe we consider gating above 0.5
        # Let's just do a naive approach: fraction of gating > 0.5
        #passed_threshold = (gating > 0.5).float()
        #fraction_selected = passed_threshold.mean()
        fraction_selected = (raw_diff > 0).float().mean()
        
        return clip_similarities, fraction_selected

    def compute_all_similarities(self, audio_feats, visual_feats):
        """
        Compute similarities for all pairs in the batch
        audio_feats:  (B, Na, D)
        visual_feats: (B, T, Nv, D)
        
        Returns:
            clip_similarities: (B, B)
            similarities: (B, B, Na, T, Nv)
            fraction_selected: float
        """
        B = audio_feats.shape[0]
        
        # Expand for all pairs
        audio_feats = audio_feats.unsqueeze(1)    # (B, 1, Na, D)
        visual_feats = visual_feats.unsqueeze(0)  # (1, B, T, Nv, D)
        
        # Normalize first, then compute
        normed_audio_feats = F.normalize(audio_feats, dim=-1)
        normed_visual_feats = F.normalize(visual_feats, dim=-1)
        
        # raw similarities
        similarities = torch.einsum('xyad,xytvd->xyatv', normed_audio_feats, normed_visual_feats)
        similarities = similarities / self.temperature
        
        # Aggregate
        clip_similarities, fraction_selected = self.aggregate_temporal_similarities(similarities)
        
        return clip_similarities, similarities, fraction_selected

    def compute_contrastive_loss(self, clip_similarities, token_sims, fraction_selected):
        """
        Compute InfoNCE-like loss with possible regularization
        clip_similarities: (B, B)
        token_sims:        (B, B, Na, T, Nv)
        fraction_selected: float
        """
        batch_size = clip_similarities.shape[0]
        labels = torch.arange(batch_size).to(clip_similarities.device)
        
        # a2v direction
        log_prob_a2v = F.log_softmax(clip_similarities, dim=1)
        losses_a2v = -log_prob_a2v[torch.arange(batch_size), labels]
        
        # v2a direction
        log_prob_v2a = F.log_softmax(clip_similarities.t(), dim=1)
        losses_v2a = -log_prob_v2a[torch.arange(batch_size), labels]
        
        # average
        contrastive_loss = (losses_a2v + losses_v2a).mean() / 2
        
        # regularization
        reg_loss = self.compute_regularization_losses(clip_similarities, token_sims)
        
        # If you want to remove the "selection_reward" that artificially pushes threshold down, you can:
        # total_loss = contrastive_loss + reg_loss
        # 
        # Alternatively, you can keep a small penalty or reward to avoid degenerate solutions.
        # For demonstration, let's remove it or just keep it extremely small:
        
        selection_reward = 0.0  # We set it to 0 now, or you can keep a tiny penalty if desired.
        
        total_loss = contrastive_loss + reg_loss + selection_reward
        
        return total_loss, contrastive_loss, reg_loss, fraction_selected, selection_reward

    def compute_regularization_losses(self, clip_sims, token_sims):
        """
        Regularization with temporal structure
        token_sims shape: [B, B, Na, T, Nv]
        """
        # 1. Non-negative pressure
        #    We clamp token_sims between [-20, 0], then penalize them if negative
        neg_sims = torch.clamp(token_sims, min=-20, max=0)
        l_nonneg = torch.mean(neg_sims ** 2)
        
        # 2. Temperature regularization
        temp_low = torch.clamp(torch.log(torch.tensor(1.0, device=token_sims.device)) - torch.log(self.temperature), min=0) ** 2
        temp_high = torch.clamp(torch.log(self.temperature) - torch.log(torch.tensor(4.0, device=token_sims.device)), min=0) ** 2
        l_cal = temp_low + temp_high
        
        # 3. Threshold regularization
        threshold_val = torch.sigmoid(self.threshold)
        l_threshold = torch.clamp(threshold_val - 0.9, min=0)**2 + torch.clamp(0.1 - threshold_val, min=0)**2
        
        # 4. Scale factor regularization
        l_scale = torch.clamp(self.scale_factor - 20.0, min=0)**2 + torch.clamp(1.0 - self.scale_factor, min=0)**2
        
        reg_loss = (0.15 * l_nonneg +
                    2.0 * l_cal +
                    0.1 * l_threshold +
                    0.1 * l_scale)
        
        return reg_loss

    def forward(self, frames, audio):
        """
        Forward pass
        Args:
            frames: (B, T, C, H, W)
            audio: (B, samples)
        """
        # 1. embed
        visual_feats = self.visual_embedder(frames)  # (B, T, Nv, D)
        audio_feats = self.audio_embedder(audio)     # (B, Na, D)
        
        if self.training:
            # compute similarities and loss
            clip_sims, token_sims, fraction_selected = self.compute_all_similarities(audio_feats, visual_feats)
            return self.compute_contrastive_loss(clip_sims, token_sims, fraction_selected)
        else:
            # inference => just compute temporal similarity matrix
            similarities = self.compute_temporal_similarity_matrix(audio_feats, visual_feats)
            return similarities

if __name__ == "__main__":
    # Sanity test
    model = AudioVisualModel()
    model.train()
    
    batch_size = 2
    num_frames = 10
    
    frames = torch.randn(batch_size, num_frames, 3, 224, 224)
    audio = torch.randn(batch_size, 16331)
    
    try:
        loss_tuple = model(frames, audio)
        print("Training forward pass success!")
        print("Loss tuple:", loss_tuple)
    except Exception as e:
        print("Error during training:", str(e))
        raise e
    
    # Test inference
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
