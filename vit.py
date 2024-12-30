import torch
import torch.nn as nn
from transformers import TimesformerModel

class ViTEmbedder(nn.Module):
    def __init__(self,):
        super().__init__()
        
        self.model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")

        self.projection = nn.Linear(768, 512)
        
        for param in self.model.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        B, T, C, H, W = x.shape
        outputs = self.model(x)
        embeddings = outputs.last_hidden_state  # (B, (T*196)+1, 768)
        embeddings = embeddings[:, 1:, :]      # (B, T*196, 768)
        
        # Reshape to separate temporal and spatial dimensions
        num_patches = 196  # 14x14 patches
        embeddings = embeddings.reshape(B, T, num_patches, 768)  # (B, T, 196, 768)
        embeddings = self.projection(embeddings)  # (B, T, 196, 512)
        return embeddings

# Test it out!
if __name__ == "__main__":
    vit = ViTEmbedder().to("cuda")
    
    # Print total number of parameters
    total_params = sum(p.numel() for p in vit.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Print model architecture
    print("\nModel layers:")
    print(vit.model)
    
    # Create dummy batch of images
    batch = torch.randn(2, 8, 3, 224, 224, device="cuda")
    
    # Get embeddings
    with torch.no_grad():
        embeddings = vit(batch)
    
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {embeddings.shape}")
    # Input shape: torch.Size([2, 10, 3, 224, 224])
    # Output shape: torch.Size([2, 10, 256, 512])