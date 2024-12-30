import torch
from torch.utils.data import Dataset, Sampler
from pathlib import Path
import numpy as np
import random
import av
from typing import Dict, List
import torch.nn as nn
import torchaudio.transforms as T
import warnings
warnings.filterwarnings("ignore")
import multiprocessing
#import dataloader
from torch.utils.data import DataLoader
# Attempt to use fork for potentially faster dataloader start
#try:
    #multiprocessing.set_start_method('fork', force=True)
#except:
multiprocessing.set_start_method('spawn', force=True)
import gc
# Global normalization constants (ImageNet)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

def extract_audio_from_video(video_path: Path) -> torch.Tensor:
    """Extract entire 1s audio from video."""
    #print("extracting audio from video")
    try:
        container = av.open(str(video_path))
        audio = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(format='s16', layout='mono', rate=16000)
        
        samples = []
        for frame in container.decode(audio):
            frame.pts = None
            frame = resampler.resample(frame)[0]
            samples.append(frame.to_ndarray().reshape(-1))
        container.close()

        samples = torch.tensor(np.concatenate(samples))
        samples = samples.float() / 32768.0  # Convert to float and normalize
        return samples
    except:
        print(f"Failed to load audio from {video_path}")
        return torch.zeros(16331)
    finally:
        #print("extracted audio from video")
        if container:
            container.close()
        #gc.collect()
        #torch.cuda.empty_cache()

def load_and_preprocess_video(video_path: str, sample_fps: int) -> torch.Tensor:
    #print("loading and preprocessing video")
    """Load 10 evenly spaced frames from the 1s video."""
    container = av.open(video_path)
    video_stream = container.streams.video[0]

    original_fps = float(video_stream.average_rate)
    video_duration = 1.0
    num_original_frames = int(round(original_fps * video_duration))
    frame_indices = np.linspace(0, num_original_frames - 1, sample_fps, dtype=int)

    frames = []
    for chosen_index in frame_indices:
        # Calculate PTS for chosen frame
        chosen_time_seconds = chosen_index / original_fps
        chosen_pts = int(chosen_time_seconds / video_stream.time_base)

        # Seek to slightly before our target
        container.seek(chosen_pts, stream=video_stream, any_frame=False, backward=True)

        # Keep track of closest frame
        closest_frame = None
        min_pts_diff = float('inf')

        # Decode frames until we find the closest one to our target PTS
        for frame in container.decode(video_stream):
            pts_diff = abs(frame.pts - chosen_pts)

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
            #print(f"  Selected frame with PTS={closest_frame.pts}") #goes upto about 12-14k
        else:
            print(f"Failed to find appropriate frame for index {chosen_index}")
            # Could potentially add a fallback here if needed
    #print("Loaded and preprocessed video")
    container.close()
    return torch.stack(frames)  # Shape: [10, 3, 224, 224]

class VideoBatchSampler(Sampler):
    def __init__(self, vid_nums: List[int], batch_size: int):
        self.vid_nums = np.array(vid_nums)
        self.batch_size = batch_size
        self.total_samples = len(vid_nums)

    def __iter__(self):
        all_indices = list(range(self.total_samples))
        random.shuffle(all_indices)
        
        current_batch = []
        used_vids = set()
        
        for idx in all_indices:
            vid = self.vid_nums[idx]
            if vid not in used_vids:
                current_batch.append(idx)
                used_vids.add(vid)
                if len(current_batch) == self.batch_size:
                    yield current_batch
                    current_batch = []
                    used_vids = set()
        
        if current_batch:
            yield current_batch
    
    def __len__(self):
        return self.total_samples // self.batch_size

class AudioVisualDataset(Dataset):
    def __init__(self, data_root: str, sample_fps: int = 8):
        self.data_root = Path(data_root)
        self.sample_fps = sample_fps
        self.video_files = sorted(list(self.data_root.glob("*.mp4")))
        
        self.vid_to_files = {}
        for file in self.video_files:
            vid_num = int(file.stem.split('_')[0])
            if vid_num not in self.vid_to_files:
                self.vid_to_files[vid_num] = []
            self.vid_to_files[vid_num].append(file)
            
        self.vid_nums = [int(f.stem.split('_')[0]) for f in self.video_files]
        print("max of vid_nums: ", max(self.vid_nums))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
    
        video_path = self.video_files[idx]
        try:
            audio = extract_audio_from_video(video_path)
            video_frame = load_and_preprocess_video(str(video_path), self.sample_fps)
            return {
                'video_path': str(video_path),
                'video_frames': video_frame, 
                'audio': audio,
                'vid_num': int(video_path.stem.split('_')[0]),
                'segment_num': int(video_path.stem.split('_')[1]),
            }
        except Exception as e:
            print(f"Error processing {self.video_files[idx]}: {str(e)}")
            return {
                'video_path': str(self.video_files[idx]),
                'video_frames': torch.zeros(8, 3, 224, 224),
                'audio': torch.zeros(16331),
                'vid_num': -1,
                'segment_num': -1
            }

def collate_fn(batch):
    # video_tokens shape will be [batch_size, 10, 3, 224, 224]
    video_tokens = torch.stack([item['video_frames'] for item in batch])
    
    max_audio_len = max(item['audio'].shape[0] for item in batch)
    audio_padded = torch.zeros(len(batch), max_audio_len)
    for i, item in enumerate(batch):
        audio_len = item['audio'].shape[0]
        audio_padded[i, :audio_len] = item['audio']
    
    return {
        'frame': video_tokens,  # Now [B, 10, 3, 224, 224]
        'audio': audio_padded,
        'vid_nums': [item['vid_num'] for item in batch],
        'segment_nums': [item['segment_num'] for item in batch],
        'video_paths': [str(item['video_path']) for item in batch]
    }

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    dataset = AudioVisualDataset(
        data_root="/home/cis/VGGSound_Splits",
        sample_fps=8
    )

    batch_sampler = VideoBatchSampler(
        vid_nums=dataset.vid_nums,
        batch_size=2  # Reduced from 48 due to 10x frames
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=1,
        #persistent_workers=True,
        #pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=2
    )

    # Quick test to visualize frames
    for batch in dataloader:
        print("Frame tensor shape:", batch['frame'].shape)  # Frame tensor shape: torch.Size([2, 10, 3, 224, 224])
        print("Audio tensor shape:", batch['audio'].shape)  # Audio tensor shape: torch.Size([2, 16331])
        
        # Visualize frames from first video in batch
        frames = batch['frame'][0]  # [10, 3, 224, 224]
        
        # Denormalize
        frames = frames * IMAGENET_STD.unsqueeze(0) + IMAGENET_MEAN.unsqueeze(0)
        
        plt.figure(figsize=(20, 4))
        for i in range(8):
            plt.subplot(1, 8, i+1)
            plt.imshow(frames[i].permute(1, 2, 0).clip(0, 1))
            plt.axis('off')
            plt.title(f'Frame {i}')
        plt.tight_layout()
        plt.savefig('frame_visualization.png')
        plt.close()
        break  # Just test one batch
