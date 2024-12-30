import torch
from torch.utils.data import DataLoader
import av
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

class AudioVisualizer:
    def __init__(self, output_resolution=(1920, 1080)):
        self.output_res = output_resolution
        self.frame_width = output_resolution[0] // 4
        self.frame_height = (output_resolution[1] - 100) // 2
        self.positions = self._calculate_frame_positions()
        
        # Create custom colormap
        colors = [
            (0,0,0,0),     # Transparent for low attention
            (0,0,1,0.5),   # Blue for medium-low
            (1,0,0,0.7),   # Red for medium-high  
            (1,1,0,1)      # Yellow for high attention
        ]
        self.cmap = LinearSegmentedColormap.from_list('custom', colors)

    def _calculate_frame_positions(self):
        positions = []
        for row in range(2):
            for col in range(4):
                x = col * self.frame_width
                y = row * self.frame_height
                positions.append((x, y))
        return positions

    def extract_frames_from_video(self, video_path: str) -> torch.Tensor:
        """Extract 10 evenly spaced frames from video, similar to dataset.py"""
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        
        print(f"\nVideo stream info:")
        print(f"Duration: {float(video_stream.duration * video_stream.time_base)} seconds")
        print(f"FPS: {float(video_stream.average_rate)}")
        print(f"Time base: {float(video_stream.time_base)}")
        print(f"Frames: {video_stream.frames}")
        
        original_fps = float(video_stream.average_rate)
        num_original_frames = int(round(original_fps * 1.0))  # 1s duration
        frame_indices = np.linspace(0, num_original_frames - 1, 8, dtype=int)
        print(f"\nCalculated frame indices: {frame_indices}")
        
        frames = []
        for chosen_index in frame_indices:
            # Calculate PTS
            chosen_time_seconds = chosen_index / original_fps
            chosen_pts = int(chosen_time_seconds / video_stream.time_base)
            
            print(f"\nFrame {len(frames)+1}/10:")
            print(f"Target index: {chosen_index}")
            print(f"Target time: {chosen_time_seconds:.3f}s")
            print(f"Target PTS: {chosen_pts}")
            
            # Seek to slightly before our target
            container.seek(chosen_pts, stream=video_stream, any_frame=False, backward=True)
            
            # Keep track of closest frame
            closest_frame = None
            min_pts_diff = float('inf')
            
            # Decode frames until we find the closest one to our target PTS
            for frame in container.decode(video_stream):
                pts_diff = abs(frame.pts - chosen_pts)
                print(f"  Got frame: PTS={frame.pts}, Time={float(frame.pts * video_stream.time_base):.3f}s, Diff={pts_diff}")
                
                if pts_diff < min_pts_diff:
                    min_pts_diff = pts_diff
                    closest_frame = frame
                
                # If we've gone too far past our target, stop
                if frame.pts > chosen_pts + original_fps/10:  # Allow 1/10th second overshoot
                    break
            
            if closest_frame is not None:
                decoded_frame = closest_frame.to_rgb().to_ndarray()
                frame_tensor = torch.from_numpy(decoded_frame).permute(2, 0, 1).float() / 255.0
                frame_tensor = torch.nn.functional.interpolate(
                    frame_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
                ).squeeze(0)
                frame_tensor = (frame_tensor - IMAGENET_MEAN) / IMAGENET_STD
                frames.append(frame_tensor)
                print(f"  Selected frame with PTS={closest_frame.pts}")
            else:
                print("  Failed to find appropriate frame!")
        
        container.close()
        return torch.stack(frames)  # [10, 3, 224, 224]  # [10, 3, 224, 224]

    def _create_frame_overlay(self, frame: np.ndarray, heatmap: np.ndarray, alpha=0.5, 
                            border_size=3, padding=10):
        """Create frame overlay with border and padding"""
        inner_width = self.frame_width - 2 * (padding + border_size)
        inner_height = self.frame_height - 2 * (padding + border_size)
        
        canvas = np.ones((self.frame_height, self.frame_width, 3), dtype=np.uint8) * 30
        
        inner_start = (padding + border_size)
        cv2.rectangle(canvas,
                     (padding, padding),
                     (self.frame_width - padding, self.frame_height - padding),
                     (255, 255, 255),
                     border_size)
        
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = np.power(heatmap, 0.5)
        
        heatmap_colored = self.cmap(heatmap)
        heatmap_bgr = (heatmap_colored[...,:3] * 255).astype(np.uint8)
        
        frame_resized = cv2.resize(frame, (inner_width, inner_height))
        heatmap_resized = cv2.resize(heatmap_bgr, (inner_width, inner_height))
        
        overlay = ((1-alpha) * frame_resized + alpha * heatmap_resized).astype(np.uint8)
        
        canvas[inner_start:inner_start + inner_height, 
               inner_start:inner_start + inner_width] = overlay
        
        return canvas

    def _draw_progress_bar(self, canvas: np.ndarray, progress: float):
        bar_height = 20
        bar_margin = 40
        bar_y = self.output_res[1] - bar_margin
        
        cv2.rectangle(canvas, 
                     (bar_margin, bar_y), 
                     (self.output_res[0] - bar_margin, bar_y + bar_height),
                     (50, 50, 50),
                     -1)
        
        width = int((self.output_res[0] - 2*bar_margin) * progress)
        cv2.rectangle(canvas,
                     (bar_margin, bar_y),
                     (bar_margin + width, bar_y + bar_height),
                     (0, 255, 0),
                     -1)

    def make_attention_video(self, model, frames, audio, output_path, video_path=None, fps=50):
        """
        Create attention visualization video
        
        Args:
            model: AudioVisual model
            frames: [1, T, C, H, W] tensor of video frames
            audio: [1, samples] tensor of audio
            output_path: Path to save output video
            video_path: Original video path for audio
        """
        model.eval()
        with torch.no_grad():
            # Get attention maps for each audio token
            visual_feats = model.visual_embedder(frames)    # [1, T, Nv, D]
            audio_feats = model.audio_embedder(audio)      # [1, Na, D]
            
            # Compute similarity matrix
            similarities = model.compute_temporal_similarity_matrix(
                audio_feats, visual_feats
            ).squeeze(0)  # [Na, T, Nv]
            
            # Process each frame
            processed_frames = []
            for i in range(frames.shape[1]):  
                frame = frames[0, i]  # [C, H, W]
                # Denormalize
                frame_np = frame.permute(1,2,0).cpu().numpy()
                frame_np = frame_np * IMAGENET_STD.squeeze(-1).squeeze(-1).numpy() + IMAGENET_MEAN.squeeze(-1).squeeze(-1).numpy()
                frame_np = np.clip(frame_np * 255, 0, 255).astype(np.uint8)
                processed_frames.append(frame_np)

            # Setup video writer
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temp_video_path = str(output_path.with_suffix('.temp.mp4'))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                temp_video_path,
                fourcc,
                fps,
                self.output_res
            )

            # Create frames
            num_audio_tokens = similarities.shape[0]
            for t in range(num_audio_tokens):
                # Create canvas
                canvas = np.ones((self.output_res[1], self.output_res[0], 3), dtype=np.uint8) * 15
                
                # Add each frame with its attention map
                for i, (x, y) in enumerate(self.positions):
                    frame = processed_frames[i]
                    attn = similarities[t, i].reshape(14, 14).cpu().numpy()
                    overlay = self._create_frame_overlay(frame, attn)
                    #print(f"Overlay shape: {overlay.shape}") #980, 640, 3 ??? 
                    #print(f"Canvas shape: {canvas.shape}") 
                    canvas[y:y+self.frame_height, x:x+self.frame_width] = overlay
                
                # Add progress bar
                self._draw_progress_bar(canvas, t / num_audio_tokens)
                
                # Convert RGB to BGR for OpenCV video writing
                canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
                writer.write(canvas_bgr)
            
            writer.release()

            # Add audio if provided
            if video_path is not None:
                import ffmpeg
                
                audio_input = ffmpeg.input(video_path).audio
                video_input = ffmpeg.input(temp_video_path).video
                
                stream = ffmpeg.output(
                    video_input, 
                    audio_input, 
                    str(output_path),
                    vcodec='copy',
                    acodec='aac'
                ).overwrite_output()
                
                try:
                    stream.run(capture_stdout=True, capture_stderr=True)
                except ffmpeg.Error as e:
                    print('stdout:', e.stdout.decode('utf8'))
                    print('stderr:', e.stderr.decode('utf8'))
                    raise e
                
                Path(temp_video_path).unlink()
            else:
                Path(temp_video_path).rename(output_path)

    def plot_attention_snapshot(self, model, frames, audio, num_timesteps=5, axes=None):
        """
        Create snapshot visualization of attention at specific timesteps

        Args:
            model: AudioVisual model
            frames: [1, T, C, H, W] tensor of video frames
            audio: [1, samples] tensor of audio
            num_timesteps: Number of audio timesteps to visualize
            axes: Optional matplotlib axes array
        """
        model.eval()
        with torch.no_grad():
            # Get attention maps for each audio token
            #print("Frames shape during plot_attention_snapshot:", frames.shape)
            #print("Audio shape during plot_attention_snapshot:", audio.shape)
            visual_feats = model.visual_embedder(frames)    # [1, T, Nv, D]
            audio_feats = model.audio_embedder(audio)      # [1, Na, D]
            
            # Compute similarity matrix
            similarities = model.compute_temporal_similarity_matrix(
                audio_feats, visual_feats
            ).squeeze(0)  # [Na, T, Nv]
            
            # Process frames
            processed_frames = []
            for i in range(frames.shape[1]):  # For each of the 10 frames
                frame = frames[0, i]  # [C, H, W]
                # Denormalize
                frame_np = frame.permute(1,2,0).cpu().numpy()
                frame_np = frame_np * IMAGENET_STD.squeeze(-1).squeeze(-1).numpy() + IMAGENET_MEAN.squeeze(-1).squeeze(-1).numpy()
                frame_np = np.clip(frame_np * 255, 0, 255).astype(np.uint8)
                processed_frames.append(frame_np)

            # Select timesteps to visualize
            num_audio_tokens = similarities.shape[0]
            selected_timesteps = np.linspace(0, num_audio_tokens-1, num_timesteps, dtype=int)
            
            if axes is None:
                fig, axes = plt.subplots(1, num_timesteps, figsize=(4*num_timesteps, 8))
                created_fig = True
            else:
                created_fig = False
                
            if num_timesteps == 1:
                axes = [axes]

            # Create visualization for each selected timestep
            for ax_idx, t in enumerate(selected_timesteps):
                # Create canvas for this timestep
                canvas = np.ones((self.output_res[1], self.output_res[0], 3), dtype=np.uint8) * 15
                
                # Add each frame with its attention map
                for i, (x, y) in enumerate(self.positions):
                    frame = processed_frames[i]
                    attn = similarities[t, i].reshape(14, 14).cpu().numpy()
                    overlay = self._create_frame_overlay(frame, attn)
                    canvas[y:y+self.frame_height, x:x+self.frame_width] = overlay
                
                # Add small progress indicator
                time_percentage = t / num_audio_tokens
                self._draw_progress_bar(canvas, time_percentage)
                
                # Display in the corresponding subplot
                axes[ax_idx].imshow(canvas)
                axes[ax_idx].set_title(f'Time: {time_percentage:.1f}s')
                axes[ax_idx].axis('off')
            
            plt.tight_layout()
            
            if created_fig:
                return fig
            return None

import torch
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    # Initialize visualizer
    visualizer = AudioVisualizer()
    
    # Path to a test video (1-second clip)
    video_path = "/home/cis/VGGSound_Splits/0_2.mp4"  # Update with your path
    
    # Extract actual frames from video
    frames = visualizer.extract_frames_from_video(video_path)  # [10, 3, 224, 224]
    frames = frames.unsqueeze(0)  # Add batch dim: [1, 10, 3, 224, 224]
    
    # Create dummy audio (not actually used for visualization, just for shape)
    dummy_audio = torch.randn(1, 16000)  # 1 second at 16kHz
    
    # Create dummy model class for testing
    class DummyModel:
        def __init__(self):
            pass
            
        def eval(self):
            return self
            
        def visual_embedder(self, frames):
            # Simulate visual embeddings [1, T, Nv, D]
            return torch.randn(1, 8, 196, 512)  # 256 = 16x16 patches
            
        def audio_embedder(self, audio):
            # Simulate audio embeddings [1, Na, D]
            return torch.randn(1, 50, 512)  # 50 audio tokens
            
        def compute_temporal_similarity_matrix(self, audio_feats, visual_feats):
            # Random similarities [1, Na, T, Nv]
            sims = torch.randn(1, 50, 8, 196)  # [batch, audio_tokens, frames, visual_tokens]
            # Make the attention patterns more structured/interesting
            sims = sims.view(1, 50, 8, 14, 14)  # Reshape to spatial layout
            
            # Create coordinate grid
            y_coords = torch.arange(14).float()
            x_coords = torch.arange(14).float()
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            # Add some gaussian blobs for more realistic looking attention
            for t in range(50):
                center_x = torch.randint(0, 14, (1,)).item()
                center_y = torch.randint(0, 14, (1,)).item()
                
                # Compute distances to center point
                distances = torch.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
                gaussian = 2.0 * torch.exp(-distances/4.0)
                
                # Add to all frames for this time step
                sims[0, t, :, :, :] += gaussian.unsqueeze(0)
            
            sims = sims.view(1, 50, 8, 196)
            return torch.sigmoid(sims)  # Normalize to [0,1]
    
    # Create dummy model
    dummy_model = DummyModel()
    
    # Test both visualization types
    print("Creating video visualization...")
    visualizer.make_attention_video(
        model=dummy_model,
        frames=frames,
        audio=dummy_audio,
        output_path="test_temporal_viz.mp4",
        video_path=video_path
    )
    
    print("\nCreating snapshot visualization...")
    fig = visualizer.plot_attention_snapshot(
        model=dummy_model,
        frames=frames,
        audio=dummy_audio,
        num_timesteps=5
    )
    plt.savefig("test_snapshot.png")
    plt.close()