from transformers import HubertModel, AutoProcessor
import torch
import torch.nn as nn

class AudioEmbedder(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        
        # Load pretrained HuBERT and processor
        self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.hubert = HubertModel.from_pretrained("ntu-spml/distilhubert")#("facebook/hubert-base-ls960") #model name: facebook/hubert-large-ls960-ft
        
        # Project HuBERT features (1024 for large model) to desired embedding dimension
        self.projection = nn.Linear(768, embedding_dim)  

        # Unfreeze the HuBERT model
        for param in self.hubert.parameters():
            param.requires_grad = True
            
        for param in self.projection.parameters():
            param.requires_grad = True
        
    def forward(self, audio_input):
        """
        Args:
            audio_input: (B, T) raw audio waveform at 16kHz
            
        Returns:
            features: (B, Na, D) where:
                B is batch size
                Na is number of audio tokens
                D is embedding_dim
        """
        # Process audio through HuBERT processor
        #print(f"Audio input shape: {audio_input.shape}")
        inputs = self.processor(
            audio_input, 
            return_tensors="pt",
            sampling_rate=16000,
            padding=True,
            return_attention_mask=True
        ).input_values.squeeze(0)
        #print(f"Inputs shape: {inputs.shape}")
        
        # Move to same device as model
        inputs = inputs.to(audio_input.device)
        
        hubert_output = self.hubert(inputs).last_hidden_state  # (B, T/320, 1024)
        
        # Project to embedding dimension
        features = self.projection(hubert_output)  # (B, T/320, embedding_dim)
        
        return features



if __name__ == "__main__":
    audio_embedder = AudioEmbedder()
    total_params = sum(p.numel() for p in audio_embedder.parameters())
    print(f"Total parameters: {total_params:,}")
    test_audio = torch.randn(1, 16331)
    print(test_audio.shape)
    features = audio_embedder(test_audio)
    print(features.shape)
    