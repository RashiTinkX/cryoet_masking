import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn.functional as F
import mrcfile
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from torchvision.transforms.functional import resize
from transformers import VideoMAEModel
import random
import pandas as pd
import seaborn as sns

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Load MRC files from directory
def load_mrc_files(folder_path):
    cryo_data_list, filenames = [], []
    for filename in os.listdir(folder_path):
        if filename.endswith(".mrc"):
            file_path = os.path.join(folder_path, filename)
            try:
                with mrcfile.open(file_path, permissive=True) as mrc:
                    cryo_data_list.append(mrc.data)
                    filenames.append(filename)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return cryo_data_list, filenames

# Preprocess data for VideoMAE model
def preprocess_for_videomae(cryo_data):
    """
    Prepares Cryo-ET data for VideoMAE model input.
    VideoMAE expects shape: (batch_size, num_frames, num_channels, height, width)
    """
    # Making sure to have enough frames
    depth = cryo_data.shape[0]
    if depth < 16:
        # Repeat frames to reach at least 16
        repeat_factor = (16 + depth - 1) // depth  # Ceiling division
        cryo_data = np.repeat(cryo_data, repeat_factor, axis=0)
    
    # Take only the first 16 frames
    cryo_data = cryo_data[:16]
    
    # Resize to 224x224
    cryo_data = torch.tensor(cryo_data, dtype=torch.float32)
    cryo_data = cryo_data.unsqueeze(1)  # Add channel dim: (D, 1, H, W)
    cryo_data = F.interpolate(cryo_data, size=(224, 224), mode='bilinear', align_corners=False)
    
    # Convert to 3 channels by repeating
    cryo_data = cryo_data.squeeze(1)  # Back to (D, H, W)
    cryo_data = cryo_data.unsqueeze(0)  # Add batch: (1, D, H, W)
    
    # Rearrange to VideoMAE expected format: (batch_size, num_frames, channels, height, width)
    cryo_data = cryo_data.permute(0, 1, 2, 3).unsqueeze(2)  # Add channel dim: (1, 16, 1, 224, 224)
    cryo_data = cryo_data.repeat(1, 1, 3, 1, 1)  # Repeat to 3 channels: (1, 16, 3, 224, 224)
    
    return cryo_data


# Create a masked version of a subtomogram for self-supervised learning
def mask_subtomogram(subtomogram, mask_ratio=0.25, patch_size=16):
    """
    Create a masked version of a subtomogram with patch-level masking.
    Fixed to handle non-standard input sizes.
    """
    if not isinstance(subtomogram, torch.Tensor):
        subtomogram = torch.tensor(subtomogram, dtype=torch.float32)
        
    B, T, C, H, W = subtomogram.shape
    
    # Ensure dimensions are multiples of patch_size
    H_patches = H // patch_size
    W_patches = W // patch_size
    
    # Handle non-divisible dimensions
    if H % patch_size != 0 or W % patch_size != 0:
        # Adjust H and W to be divisible by patch_size
        new_H = H_patches * patch_size
        new_W = W_patches * patch_size
        
        # Resize subtomogram
        subtomogram = F.interpolate(
            subtomogram.view(B*T, C, H, W),
            size=(new_H, new_W),
            mode='bilinear',
            align_corners=False
        ).view(B, T, C, new_H, new_W)
        
        # Update dimensions
        H, W = new_H, new_W
        H_patches = H // patch_size
        W_patches = W // patch_size
    
    # Calculate number of patches per frame
    num_patches_per_frame = H_patches * W_patches
    
    # Total patches across all frames
    total_patches = T * num_patches_per_frame
    
    # Calculate number of patches to mask
    num_masked = int(total_patches * mask_ratio)
    
    # Create binary mask (1 = keep, 0 = mask)
    patch_mask = torch.ones(B, T, num_patches_per_frame)
    
    # Randomly select indices to mask
    for b in range(B):
        # Flatten to select from all frames
        flat_indices = torch.randperm(total_patches)[:num_masked]
        
        # Convert to frame and patch indices
        frame_indices = flat_indices // num_patches_per_frame
        patch_indices = flat_indices % num_patches_per_frame
        
        # Set selected patches to be masked (0)
        for f_idx, p_idx in zip(frame_indices, patch_indices):
            if f_idx < T:  # Ensure we're within bounds
                patch_mask[b, f_idx, p_idx] = 0
    
    # Reshape subtomogram to patches
    x_patches = subtomogram.reshape(B, T, C, H_patches, patch_size, W_patches, patch_size)
    x_patches = x_patches.permute(0, 1, 4, 6, 2, 3, 5)
    x_patches = x_patches.reshape(B, T, patch_size * patch_size * C, H_patches, W_patches)
    
    # Apply mask to patches (expanding mask to match patch dimensions)
    expanded_mask = patch_mask.reshape(B, T, 1, 1, num_patches_per_frame)
    expanded_mask = expanded_mask.reshape(B, T, 1, H_patches, W_patches)
    expanded_mask = expanded_mask.repeat(1, 1, patch_size * patch_size * C, 1, 1)
    
    # Apply mask (multiply to keep unmasked patches, zero out masked patches)
    masked_patches = x_patches * expanded_mask
    
    # Reshape back to original format
    masked_volume = masked_patches.reshape(B, T, patch_size, patch_size, C, H_patches, W_patches)
    masked_volume = masked_volume.permute(0, 1, 4, 5, 2, 6, 3)
    masked_volume = masked_volume.reshape(B, T, C, H, W)
    
    # For visualization, create a simplified mask showing which patches are masked
    vis_mask = patch_mask.reshape(B, T, H_patches, W_patches)
    vis_mask = vis_mask.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)
    
    return masked_volume, vis_mask

# Initialize VideoMAE model
def initialize_videomae_model(checkpoint_path=None):
    # Load model from Hugging Face
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

    # Load custom checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    
    # Set to evaluation mode
    model.eval()
    return model

# Apply VideoMAE to a single subtomogram for reconstruction
def apply_videomae(model, subtomogram, mask_ratio=0.25):
    with torch.no_grad():
        # Preprocess
        processed = preprocess_for_videomae(subtomogram)
        
        # Create masked version with patch-level masking
        masked_volume, mask = mask_subtomogram(processed, mask_ratio)
        
        # Get model output
        output = model(masked_volume)
        hidden_state = output.last_hidden_state
        
        # Get dimensions
        batch_size, seq_length, hidden_dim = hidden_state.shape
        
        # Standard VideoMAE has a sequence length of 1568 for 16 frames with 224x224 resolution
        # Each frame should have (224/16)×(224/16) = 14×14 = 196 patches
        # So seq_length should be 16×196 = 3136, but it's often smaller due to padding tokens
        
        # Dynamically calculate patches per frame
        frames = 16  # Standard for VideoMAE
        patches_per_frame = seq_length // frames
        patch_size = 16  # Standard VideoMAE patch size
        
        # Calculate the spatial dimensions based on patches_per_frame
        patches_per_side = int(np.sqrt(patches_per_frame))
        
        # Add safety check to ensure we have a perfect square
        if patches_per_side**2 != patches_per_frame:
            print(f"Warning: patches_per_frame ({patches_per_frame}) is not a perfect square.")
            # Find the closest perfect square
            patches_per_side = int(np.sqrt(patches_per_frame))
            patches_per_frame = patches_per_side**2
            print(f"Adjusting to {patches_per_frame} patches per frame ({patches_per_side}×{patches_per_side}).")
        
        # Now reshape safely
        try:
            # Reshape to batch x frames x patches_per_side x patches_per_side x hidden_dim
            reconstructed = hidden_state.reshape(
                batch_size, frames, patches_per_side, patches_per_side, hidden_dim
            )
            
            # Simple feature extraction - use first few dimensions of hidden state
            feature_dim = min(3, hidden_dim)  # Take up to 3 channels (RGB)
            reconstructed = reconstructed[:, :, :, :, :feature_dim]
            
            # Unpatchify - reconstruct the spatial dimensions
            height = width = patches_per_side * patch_size
            
            # Reshape to get back image-like structure
            reconstructed = reconstructed.permute(0, 1, 4, 2, 3)  # B, T, C, H, W in patches
            
            # Create coordinates for each patch
            batch_size, frames, channels, h_patches, w_patches = reconstructed.shape
            reconstructed_full = torch.zeros(batch_size, frames, channels, 
                                           h_patches*patch_size, w_patches*patch_size, 
                                           device=reconstructed.device)
            
            # Fill in patch by patch
            for h in range(h_patches):
                for w in range(w_patches):
                    h_start = h * patch_size
                    w_start = w * patch_size
                    # Broadcast the patch values to fill patch_size×patch_size area
                    patch_values = reconstructed[:, :, :, h, w].unsqueeze(3).unsqueeze(4)
                    patch_values = patch_values.expand(-1, -1, -1, patch_size, patch_size)
                    reconstructed_full[:, :, :, h_start:h_start+patch_size, 
                                     w_start:w_start+patch_size] = patch_values
                    
            # Convert RGB to grayscale if needed
            if feature_dim >= 3:
                # RGB to grayscale conversion
                reconstructed_gray = 0.299 * reconstructed_full[:, :, 0] + \
                                     0.587 * reconstructed_full[:, :, 1] + \
                                     0.114 * reconstructed_full[:, :, 2]
                reconstructed_full = reconstructed_gray.unsqueeze(2)  # Add channel dimension back
            
            # Normalize to 0-1 range
            reconstructed_full = (reconstructed_full - reconstructed_full.min()) / \
                              (reconstructed_full.max() - reconstructed_full.min() + 1e-6)
            
        except RuntimeError as e:
            print(f"Reshape error: {e}")
            print(f"Dims: batch_size={batch_size}, seq_length={seq_length}, hidden_dim={hidden_dim}")
            print(f"Calculated: frames={frames}, patches_per_frame={patches_per_frame}, patches_per_side={patches_per_side}")
            
            # Fallback reconstruction: just use raw output and resize
            print("Using fallback reconstruction method")
            
            # Take the first few dimensions as features
            feature_dim = min(3, hidden_dim)
            features = hidden_state[:, :, :feature_dim]
            
            # Normalize features
            features = (features - features.min()) / (features.max() - features.min() + 1e-6)
            
            # Reshape to frames
            features_frames = features.reshape(batch_size, frames, -1, feature_dim)
            
            # Interpolate to create images
            height = width = 224  # Standard VideoMAE input size
            reconstructed_full = torch.zeros(batch_size, frames, feature_dim, height, width)
            
            for f in range(frames):
                frame_features = features_frames[:, f, :, :]  # B, N, C
                # Create a grid of values to be reshaped into an image
                frame_length = frame_features.shape[1]
                side_length = int(np.sqrt(frame_length))
                
                if side_length**2 != frame_length:
                    # If not perfect square, pad
                    side_length = int(np.ceil(np.sqrt(frame_length)))
                    padding_needed = side_length**2 - frame_length
                    # Pad with zeros
                    padding = torch.zeros(batch_size, padding_needed, feature_dim, 
                                         device=frame_features.device)
                    frame_features = torch.cat([frame_features, padding], dim=1)
                
                # Reshape to 2D grid
                frame_grid = frame_features.reshape(batch_size, side_length, side_length, feature_dim)
                frame_grid = frame_grid.permute(0, 3, 1, 2)  # B, C, H, W
                
                # Resize to full resolution
                frame_full = F.interpolate(frame_grid, size=(height, width), 
                                         mode='bilinear', align_corners=False)
                
                reconstructed_full[:, f, :, :, :] = frame_full
            
            # Convert RGB to grayscale if needed
            if feature_dim >= 3:
                reconstructed_gray = 0.299 * reconstructed_full[:, :, 0] + \
                                   0.587 * reconstructed_full[:, :, 1] + \
                                   0.114 * reconstructed_full[:, :, 2]
                reconstructed_full = reconstructed_gray.unsqueeze(2)
            
            # Normalize to 0-1 range
            reconstructed_full = (reconstructed_full - reconstructed_full.min()) / \
                              (reconstructed_full.max() - reconstructed_full.min() + 1e-6)
        
    return processed, masked_volume, reconstructed_full, mask

# Evaluate reconstruction quality
def evaluate_reconstruction(original, reconstructed):
    """
    Evaluate reconstruction quality with enhanced error handling.
    """
    # Convert tensors to numpy if needed
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()
    
    # Get to 2D representations for comparison
    # For volumes, take the average across frames
    if original.ndim > 2:
        # Handle different dimensions
        if original.ndim == 3:  # [D, H, W]
            original = np.mean(original, axis=0)
        elif original.ndim == 4:  # [D, C, H, W]
            if original.shape[1] == 1:  # Single channel
                original = np.mean(original[:, 0], axis=0)
            else:  # Multiple channels
                # Convert to grayscale first
                gray = 0.299 * original[:, 0] + 0.587 * original[:, 1] + 0.114 * original[:, min(2, original.shape[1]-1)]
                original = np.mean(gray, axis=0)
        elif original.ndim == 5:  # [B, D, C, H, W]
            if original.shape[2] == 1:  # Single channel
                original = np.mean(original[0, :, 0], axis=0)
            else:  # Multiple channels
                gray = 0.299 * original[0, :, 0] + 0.587 * original[0, :, 1] + 0.114 * original[0, :, min(2, original.shape[2]-1)]
                original = np.mean(gray, axis=0)
    
    # Similar processing for reconstructed
    if reconstructed.ndim > 2:
        if reconstructed.ndim == 3:
            reconstructed = np.mean(reconstructed, axis=0)
        elif reconstructed.ndim == 4:
            if reconstructed.shape[1] == 1:
                reconstructed = np.mean(reconstructed[:, 0], axis=0)
            else:
                gray = 0.299 * reconstructed[:, 0] + 0.587 * reconstructed[:, 1] + 0.114 * reconstructed[:, min(2, reconstructed.shape[1]-1)]
                reconstructed = np.mean(gray, axis=0)
        elif reconstructed.ndim == 5:
            if reconstructed.shape[2] == 1:
                reconstructed = np.mean(reconstructed[0, :, 0], axis=0)
            else:
                gray = 0.299 * reconstructed[0, :, 0] + 0.587 * reconstructed[0, :, 1] + 0.114 * reconstructed[0, :, min(2, reconstructed.shape[2]-1)]
                reconstructed = np.mean(gray, axis=0)
    
    # Ensure they have the same shape
    if original.shape != reconstructed.shape:
        print(f"Shape mismatch: original {original.shape}, reconstructed {reconstructed.shape}")
        # Resize reconstructed to match original
        reconstructed = resize(torch.tensor(reconstructed).unsqueeze(0), 
                              size=original.shape,
                              antialias=True).squeeze().numpy()
    
    # Handle NaN or Inf values
    original = np.nan_to_num(original)
    reconstructed = np.nan_to_num(reconstructed)
    
    # Normalize values to 0-1 range if needed
    if original.max() > 1.0 or original.min() < 0.0:
        original = (original - original.min()) / (original.max() - original.min() + 1e-8)
    if reconstructed.max() > 1.0 or reconstructed.min() < 0.0:
        reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min() + 1e-8)
    
    # Compute metrics
    try:
        psnr_val = psnr(original, reconstructed, data_range=1.0)
        ssim_val = ssim(original, reconstructed, data_range=1.0)
        return psnr_val, ssim_val
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return 0, 0

# Extract additional features from each cryo-ET volume
def extract_volume_features(volume):
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy()
        
    # Simple statistical features
    features = {
        'mean': np.mean(volume),
        'std': np.std(volume),
        'min': np.min(volume),
        'max': np.max(volume),
        'median': np.median(volume),
        'volume_size': volume.size,
        'dynamic_range': np.max(volume) - np.min(volume)
    }
    
    return features

# Helper function to consistently extract the same view
def get_consistent_view(data, frame_idx=0):
    """
    Extract a consistent 2D view from data tensors regardless of dimensions.
    
    Args:
        data: Input data tensor (torch.Tensor or numpy array)
        frame_idx: Frame index to extract
        
    Returns:
        2D numpy array for visualization
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    if data.ndim == 5:  # [B, T, C, H, W]
        if data.shape[2] == 1:  # Single channel
            view = data[0, frame_idx, 0]
        elif data.shape[2] == 3:  # RGB
            view = np.transpose(data[0, frame_idx], (1, 2, 0))
            # Convert to grayscale if needed
            if view.ndim == 3 and view.shape[2] == 3:
                view = 0.299 * view[:, :, 0] + 0.587 * view[:, :, 1] + 0.114 * view[:, :, 2]
        else:  # Multi-channel, not RGB
            view = data[0, frame_idx, 0]
    elif data.ndim == 4:  # [T, C, H, W] or [B, H, W, C]
        if data.shape[1] == 1 or data.shape[1] == 3:  # Likely [T, C, H, W]
            if data.shape[1] == 1:
                view = data[frame_idx, 0]
            else:
                view = np.transpose(data[frame_idx], (1, 2, 0))
                # Convert to grayscale if needed
                if view.ndim == 3 and view.shape[2] == 3:
                    view = 0.299 * view[:, :, 0] + 0.587 * view[:, :, 1] + 0.114 * view[:, :, 2]
        else:  # Might be [B, H, W, C]
            view = data[0] if data.shape[0] == 1 else data[frame_idx]
    elif data.ndim == 3:  # [T, H, W]
        view = data[frame_idx]
    else:  # [H, W]
        view = data
    
    # Ensure the output is 2D
    if view.ndim > 2:
        view = np.mean(view, axis=2)
    
    return view

# Visualize original, masked and reconstructed data
def visualize_reconstruction(original, masked, reconstructed, mask=None, metrics=None, frame_idx=0):
    """
    Visualize original, masked and reconstructed data with enhanced error handling.
    """
    # Use the consistent view extraction function
    original_view = get_consistent_view(original, frame_idx)
    masked_view = get_consistent_view(masked, frame_idx)
    reconstructed_view = get_consistent_view(reconstructed, frame_idx)
    
    # For mask (if provided)
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        
        if mask.ndim == 4:  # [B, T, H, W]
            if frame_idx >= mask.shape[1]:
                frame_idx = 0
            mask_view = mask[0, frame_idx]
        elif mask.ndim == 3:  # [T, H, W]
            if frame_idx >= mask.shape[0]:
                frame_idx = 0
            mask_view = mask[frame_idx]
        else:
            mask_view = mask
    else:
        mask_view = None
    
    # Create figure
    fig, axes = plt.subplots(1, 4 if mask_view is not None else 3, figsize=(20, 6))
    
    # Plot original
    axes[0].imshow(original_view, cmap='gray')
    axes[0].set_title("Original (Frame {})".format(frame_idx), fontsize=14)
    axes[0].axis('off')
    
    # Plot masked
    axes[1].imshow(masked_view, cmap='gray')
    axes[1].set_title("Masked Input (Frame {})".format(frame_idx), fontsize=14)
    axes[1].axis('off')
    
    # Plot reconstructed with metrics
    axes[2].imshow(reconstructed_view, cmap='gray')
    if metrics:
        axes[2].set_title(f"Reconstructed (PSNR: {metrics[0]:.2f}, SSIM: {metrics[1]:.2f})", fontsize=14)
    else:
        axes[2].set_title("Reconstructed (Frame {})".format(frame_idx), fontsize=14)
    axes[2].axis('off')
    
    # Plot mask overlay if provided
    if mask_view is not None:
        # Create an RGB image from the original with red overlay for masked regions
        axes[3].set_title("Mask Overlay (Red = Masked)", fontsize=14)
        
        # Create a 3-channel version of the original view
        img = np.stack([original_view] * 3, axis=2)
        
        # Normalize to 0-1 if needed
        if img.max() > 1.0:
            img = img / 255.0
        
        # Process mask overlay
        # If most values are high, invert the mask (assuming 1 = keep, 0 = mask)
        if mask_view.max() <= 1 and mask_view.min() >= 0:
            if np.mean(mask_view) > 0.5:
                mask_view = 1 - mask_view  # Invert
        
        # Create a red mask overlay
        img_overlay = img.copy()
        
        # Apply red tint to masked areas
        red_tint = np.zeros_like(img_overlay)
        red_tint[:, :, 0] = 1.0  # Full red
        red_tint[:, :, 1] = 0.3  # Some green
        red_tint[:, :, 2] = 0.3  # Some blue
        
        # Create binary mask for overlay
        binary_mask = (mask_view < 0.5).astype(float)
        binary_mask_3d = np.stack([binary_mask] * 3, axis=2)
        
        # Blend using the mask
        img_overlay = img_overlay * (1 - binary_mask_3d) + red_tint * binary_mask_3d
        
        axes[3].imshow(img_overlay)
        axes[3].set_title("Mask Overlay (Red = Masked)")
        axes[3].axis('off')
        
    plt.subplots_adjust(top=0.85)
    plt.tight_layout()
    return fig

# Create comparison plot of all MRC files
def create_comparison_plot(filenames, metrics_data, features_data=None):
    df = pd.DataFrame({
        'Filename': filenames,
        'PSNR': [m[0] for m in metrics_data],
        'SSIM': [m[1] for m in metrics_data]
    })
    
    # Add features if available
    if features_data:
        for feature_name in features_data[0].keys():
            df[feature_name] = [f[feature_name] for f in features_data]
    
    # Sort by PSNR for better visualization
    df = df.sort_values('PSNR', ascending=False)
    
    # Create figure for comparing all files
    plt.figure(figsize=(14, 10))
    
    # Plot 1: PSNR and SSIM comparison
    plt.subplot(2, 1, 1)
    sns.barplot(x='Filename', y='PSNR', data=df)
    plt.xticks(rotation=90)
    plt.title('PSNR Comparison Across All MRC Files')
    plt.tight_layout()
    
    # Force save the figure
    psnr_fig_path = os.path.join(os.getcwd(), "psnr_comparison.png")
    plt.savefig(psnr_fig_path)
    print(f"Saved PSNR plot to: {psnr_fig_path}")
    
    # Create new figure for SSIM
    plt.figure(figsize=(14, 10))
    plt.subplot(1, 1, 1)
    sns.barplot(x='Filename', y='SSIM', data=df)
    plt.xticks(rotation=90)
    plt.title('SSIM Comparison Across All MRC Files')
    plt.tight_layout()
    
    # Force save the figure
    ssim_fig_path = os.path.join(os.getcwd(), "ssim_comparison.png")
    plt.savefig(ssim_fig_path)
    print(f"Saved SSIM plot to: {ssim_fig_path}")
    
    # Create additional plots if we have features
    if features_data:
        # Create a correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_cols = ['PSNR', 'SSIM'] + list(features_data[0].keys())
        correlation_df = df[correlation_cols]
        correlation_matrix = correlation_df.corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Between Metrics and Volume Features')
        plt.tight_layout()
        
        # Force save the figure
        corr_fig_path = os.path.join(os.getcwd(), "correlation_heatmap.png")
        plt.savefig(corr_fig_path)
        print(f"Saved correlation heatmap to: {corr_fig_path}")
        
        # Create a scatter plot of PSNR vs each main feature
        plt.figure(figsize=(16, 10))
        feature_plots = min(len(features_data[0]), 4)  # Limit to top 4 features
        for i, feature in enumerate(list(features_data[0].keys())[:feature_plots]):
            plt.subplot(2, 2, i+1)
            sns.scatterplot(x=feature, y='PSNR', data=df)
            plt.title(f'PSNR vs {feature}')
        plt.tight_layout()
        
        # Force save the figure
        scatter_fig_path = os.path.join(os.getcwd(), "psnr_vs_features.png")
        plt.savefig(scatter_fig_path)
        print(f"Saved PSNR vs Features plot to: {scatter_fig_path}")
    
    return df

# Main function with VideoMAE implementation
def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Define paths
    data_folder = r"C:\Users\Rashi\OneDrive\Desktop\IMP\projects\CMU\cryo-ET-samples"
    videomae_checkpoint_path = r"C:\Users\Rashi\Downloads\videomae_checkpoint.pth"
    output_folder = os.path.join(r"C:\Users\Rashi\OneDrive\Desktop\IMP\projects\CMU", "results")    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize model
    model = initialize_videomae_model(videomae_checkpoint_path)
    
    # Load data
    cryo_data_list, filenames = load_mrc_files(data_folder)
    print(f"Loaded {len(cryo_data_list)} .mrc files")
    
    if not cryo_data_list:
        print("No data loaded! Please check the path.")
        return
    
    # Process a sample for demonstration
    sample_idx = 0  # Use first sample
    if len(cryo_data_list) > 0:
        print(f"Processing sample demonstration with {filenames[sample_idx]}...")
        
        # Apply VideoMAE to the sample
        original, masked, reconstructed, mask = apply_videomae(
            model, 
            cryo_data_list[sample_idx],
            mask_ratio=0.25  # Using a higher mask ratio for better learning
        )
        
        # Evaluate reconstruction
        # Use the first frame for demonstration
        frame_idx = 0
        psnr_val, ssim_val = evaluate_reconstruction(
            original.squeeze(0)[frame_idx].mean(dim=0) if original.dim() > 4 else original.squeeze(0)[frame_idx], 
            reconstructed.squeeze(0)[frame_idx].mean(dim=0) if reconstructed.dim() > 4 else reconstructed.squeeze(0)[frame_idx]
        )
        print(f"Sample reconstruction - PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
        
        # Visualize results with visualization
        for frame_idx in range(min(3, original.shape[1])):  # Show first 3 frames
            fig = visualize_reconstruction(
                original, 
                masked, 
                reconstructed, 
                mask,
                (psnr_val, ssim_val),
                frame_idx=frame_idx
            )
            plt.savefig(os.path.join(output_folder, f"sample_reconstruction_frame{frame_idx}.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        # Process all samples
        print("Processing all samples...")
        all_metrics = []
        all_features = []
        
        for i, data in enumerate(cryo_data_list):
            print(f"Processing sample {i+1}/{len(cryo_data_list)}: {filenames[i]}")
            
            # Extract features before preprocessing
            features = extract_volume_features(data)
            all_features.append(features)
            
            # Apply VideoMAE
            original, masked, reconstructed, mask = apply_videomae(model, data, mask_ratio=0.25)
            
            # Calculate metrics across all frames
            frame_metrics = []
            for frame_idx in range(min(original.shape[1], 5)):  # Limit to first 5 frames for efficiency
                metrics = evaluate_reconstruction(
                    original.squeeze(0)[frame_idx].mean(dim=0) if original.dim() > 4 else original.squeeze(0)[frame_idx],
                    reconstructed.squeeze(0)[frame_idx].mean(dim=0) if reconstructed.dim() > 4 else reconstructed.squeeze(0)[frame_idx]
                )
                frame_metrics.append(metrics)
            
            # Average metrics across frames
            avg_psnr = np.mean([m[0] for m in frame_metrics])
            avg_ssim = np.mean([m[1] for m in frame_metrics])
            all_metrics.append((avg_psnr, avg_ssim))
            
            # Create detailed visualization for each sample
            fig = visualize_reconstruction(
                original, 
                masked, 
                reconstructed, 
                mask,
                (avg_psnr, avg_ssim),
                frame_idx=0  # Use first frame for overview
            )
            
            sample_name = filenames[i].split('.')[0]
            plt.savefig(os.path.join(output_folder, f"reconstruction_{sample_name}.png"))
            plt.close(fig)
        
        # Report average metrics
        avg_psnr = np.mean([m[0] for m in all_metrics])
        avg_ssim = np.mean([m[1] for m in all_metrics])
        print(f"Average PSNR: {avg_psnr:.2f}, Average SSIM: {avg_ssim:.4f}")
        
        # Create comparison plot
        print("Creating comparison plots...")
        df = create_comparison_plot(
            [f.split('.')[0] for f in filenames],  # Remove .mrc extension
            all_metrics,
            all_features
        )
        
        # Save comparison data
        csv_path = os.path.join(output_folder, "cryo_et_comparison_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved metrics CSV to: {csv_path}")
        
        # Create top sample visualizations
        # top 6 samples by PSNR
        top_indices = np.argsort([m[0] for m in all_metrics])[-6:]
        print("Creating detailed reconstructions for top 6 samples...")
        
        # this creates montage of top samples
        plt.figure(figsize=(20, 10))
        
        # Helper function to consistently extract the same view
        def get_consistent_view(data, frame_idx=0):
            """
            Extract a consistent 2D view from data tensors regardless of dimensions.
            
            Args:
                data: Input data tensor (torch.Tensor or numpy array)
                frame_idx: Frame index to extract
                
            Returns:
                2D numpy array for visualization
            """
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            
            if data.ndim == 5:  # [B, T, C, H, W]
                if data.shape[2] == 1:  # Single channel
                    view = data[0, frame_idx, 0]
                elif data.shape[2] == 3:  # RGB
                    view = np.transpose(data[0, frame_idx], (1, 2, 0))
                    # Convert to grayscale if needed
                    if view.ndim == 3 and view.shape[2] == 3:
                        view = 0.299 * view[:, :, 0] + 0.587 * view[:, :, 1] + 0.114 * view[:, :, 2]
                else:  # Multi-channel, not RGB
                    view = data[0, frame_idx, 0]
            elif data.ndim == 4:  # [T, C, H, W] or [B, H, W, C]
                if data.shape[1] == 1 or data.shape[1] == 3:  # Likely [T, C, H, W]
                    if data.shape[1] == 1:
                        view = data[frame_idx, 0]
                    else:
                        view = np.transpose(data[frame_idx], (1, 2, 0))
                        # Convert to grayscale if needed
                        if view.ndim == 3 and view.shape[2] == 3:
                            view = 0.299 * view[:, :, 0] + 0.587 * view[:, :, 1] + 0.114 * view[:, :, 2]
                else:  # Might be [B, H, W, C]
                    view = data[0] if data.shape[0] == 1 else data[frame_idx]
            elif data.ndim == 3:  # [T, H, W]
                view = data[frame_idx]
            else:  # [H, W]
                view = data
            
            # Ensure the output is 2D
            if view.ndim > 2:
                view = np.mean(view, axis=2)
            
            return view
        
        for i, idx in enumerate(top_indices):
            plt.subplot(2, 3, i+1)
            sample_name = filenames[idx].split('.')[0]
            
            # Re-process top samples for visualization
            original, masked, reconstructed, mask = apply_videomae(model, cryo_data_list[idx], mask_ratio=0.25)
            
            # Get frame to display
            display_frame = 0
            
            # Use the consistent view extraction function for both visualizations
            orig_frame = get_consistent_view(original, display_frame)
            recon_frame = get_consistent_view(reconstructed, display_frame)
            
            # Create side-by-side comparison
            combined = np.concatenate([orig_frame, recon_frame], axis=1)
            plt.imshow(combined, cmap='gray')
            plt.title(f"{sample_name} (PSNR: {all_metrics[idx][0]:.2f})")
            plt.axis('off')
        
        plt.tight_layout()
        top_samples_path = os.path.join(output_folder, "top_samples_comparison.png")
        plt.savefig(top_samples_path)
        plt.close()
        
        # Create a video-style visualization for the best sample
        best_sample_idx = top_indices[-1]
        original, masked, reconstructed, mask = apply_videomae(
            model, 
            cryo_data_list[best_sample_idx], 
            mask_ratio=0.25
        )
        
        # Generate frame-by-frame visualization
        num_frames = min(16, original.shape[1])
        plt.figure(figsize=(16, 16))
        
        for frame_idx in range(num_frames):
            plt.subplot(4, 4, frame_idx+1)
            
            # Use the consistent view extraction function for frame-by-frame comparison
            orig_frame = get_consistent_view(original, frame_idx)
            recon_frame = get_consistent_view(reconstructed, frame_idx)
            
            combined = np.concatenate([orig_frame, recon_frame], axis=1)
            plt.imshow(combined, cmap='gray')
            plt.title(f"Frame {frame_idx}")
            plt.axis('off')
        
        plt.suptitle(f"Frame-by-Frame Comparison for {filenames[best_sample_idx]}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        frame_comparison_path = os.path.join(output_folder, "best_sample_frame_comparison.png")
        plt.savefig(frame_comparison_path)
        plt.close()
        
        # Create a visualization of the masking process
        plt.figure(figsize=(15, 5))
        
        # Choose a sample frame from the best sample
        display_frame = 0
        
        # Use the consistent view extraction function for masking visualization
        orig_frame = get_consistent_view(original, display_frame)
        masked_frame = get_consistent_view(masked, display_frame)
        mask_frame = mask.squeeze(0)[display_frame].cpu().numpy() if isinstance(mask, torch.Tensor) else mask[0, display_frame]
        
        # Plot original
        plt.subplot(1, 3, 1)
        plt.imshow(orig_frame, cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        # Plot mask visualization
        plt.subplot(1, 3, 2)
        plt.imshow(mask_frame, cmap='gray')
        plt.title("Mask (White = Visible, Black = Masked)")
        plt.axis('off')
        
        # Plot masked input
        plt.subplot(1, 3, 3)
        plt.imshow(masked_frame, cmap='gray')
        plt.title("Masked Input")
        plt.axis('off')
        
        plt.tight_layout()
        masking_viz_path = os.path.join(output_folder, "masking_visualization.png")
        plt.savefig(masking_viz_path)
        plt.close()
        
        print("Analysis complete! Files saved to the following location:")
        print(f"Output directory: {output_folder}")
        print("Check for the following files:")
        print("- sample_reconstruction_frame*.png")
        print("- reconstruction_*.png for each sample")
        print("- cryo_et_comparison_metrics.csv")
        print("- top_samples_comparison.png")
        print("- best_sample_frame_comparison.png")
        print("- masking_visualization.png")
        print("- Additional comparison plots from create_comparison_plot function")

if __name__ == "__main__":
    main()