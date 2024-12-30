import torch
import torch.nn as nn
import timm

class ViTEmbedder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14') #torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

        self.projection = nn.Linear(384, 512)
        
        for param in self.model.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        B, T, C, H, W = x.shape
        # Reshape to treat each frame as a separate batch item
        x = x.reshape(B * T, C, H, W)
        # Process through ViT
        embeddings = self.model.get_intermediate_layers(x, n=1)[0]
        embeddings = self.projection(embeddings)
        # Reshape back to separate batch and time dimensions
        embeddings = embeddings.reshape(B, T, embeddings.shape[1], -1)
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
    batch = torch.randn(2, 10, 3, 224, 224, device="cuda")
    
    # Get embeddings
    with torch.no_grad():
        embeddings = vit(batch)
    
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {embeddings.shape}")
    # Input shape: torch.Size([2, 10, 3, 224, 224])
    # Output shape: torch.Size([2, 10, 256, 512])