import os
import numpy as np
import torch
import torch.nn.functional as F
import mrcfile
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Import ResNet-34 from Kensho Hara's repo
from models import resnet  

# Import VideoMAE from Hugging Face
from transformers import VideoMAEFeatureExtractor, VideoMAEModel
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
import random

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # If using GPU

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)


# Path to Cryo-ET data folder
data_folder = r"C:\Users\Rashi\Downloads\cryo-ET-samples"

# Function to load all .mrc files
def load_mrc_files(folder_path):
    cryo_data_list, filenames = [], []
    for filename in os.listdir(folder_path):
        if filename.endswith(".mrc"):
            file_path = os.path.join(folder_path, filename)
            with mrcfile.open(file_path, permissive=True) as mrc:
                cryo_data_list.append(mrc.data)  # 3D volume data
                filenames.append(filename)
    return cryo_data_list, filenames

# Load all Cryo-ET data
cryo_data_list, filenames = load_mrc_files(data_folder)
print(f"Loaded {len(cryo_data_list)} .mrc files")

# Function to preprocess data for ResNet34
def preprocess_for_resnet(cryo_data):
    cryo_data = np.expand_dims(cryo_data, axis=0)  # (1, D, H, W)
    cryo_data = np.repeat(cryo_data, 3, axis=0)  # Convert to 3-channel (3, D, H, W)
    cryo_data = np.expand_dims(cryo_data, axis=0)  # Add batch dimension (1, 3, D, H, W)
    cryo_data = torch.tensor(cryo_data, dtype=torch.float32)  
    cryo_data = F.interpolate(cryo_data, size=(32, 112, 112), mode='trilinear', align_corners=False)
    return cryo_data

# Function to preprocess data for VideoMAE
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

# Process Cryo-ET data for both models
preprocessed_resnet_data = [preprocess_for_resnet(cryo) for cryo in cryo_data_list]
preprocessed_videomae_data = [preprocess_for_videomae(cryo) for cryo in cryo_data_list]

print(f"Preprocessed {len(preprocessed_resnet_data)} samples for model input.")


# Load 3D-ResNet34 Model
model_resnet = resnet.resnet34(sample_size=112, sample_duration=16, num_classes=400)
checkpoint_path = "resnet-34-kinetics-cpu.pth"
checkpoint = torch.load(checkpoint_path, map_location="cpu")
model_resnet.load_state_dict(checkpoint["state_dict"], strict=False)

# Modify the model to remove the classification layer
model_resnet.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
model_resnet.fc = torch.nn.Identity()
model_resnet.eval()
print("3D-ResNet34 model loaded successfully!")

# Load Pre-trained VideoMAE Model from checkpoint
videomae_checkpoint_path = r"C:\Users\Rashi\Downloads\videomae_checkpoint.pth"

# Load VideoMAE model architecture from Hugging Face
feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")
model_videomae = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")

# Load pretrained weights into VideoMAE
videomae_checkpoint = torch.load(videomae_checkpoint_path, map_location="cpu")
model_videomae.load_state_dict(videomae_checkpoint, strict=False)  # Allow partial loading if necessary

# Set model to evaluation mode
model_videomae.eval()
print("VideoMAE model loaded successfully!")

# Extract Features Using 3D-ResNet34
features_resnet = []
for i, cryo_tensor in enumerate(preprocessed_resnet_data):
    with torch.no_grad():
        feature_vector = model_resnet(cryo_tensor)
        features_resnet.append(feature_vector.numpy().flatten())

features_resnet = np.array(features_resnet)
print(f"Extracted features from 3D-ResNet34: {features_resnet.shape}")

# Extract Features Using VideoMAE
features_videomae = []
for i, cryo_tensor in enumerate(preprocessed_videomae_data):
    with torch.no_grad():
        outputs = model_videomae(cryo_tensor)
        feature_vector = outputs.last_hidden_state.mean(dim=1)
        features_videomae.append(feature_vector.numpy().flatten())

features_videomae = np.array(features_videomae)
print(f"Extracted features from VideoMAE: {features_videomae.shape}")

# Apply t-SNE for Visualization
tsne = TSNE(n_components=2, random_state=SEED)
features_2d_resnet = tsne.fit_transform(features_resnet)
features_2d_videomae = tsne.fit_transform(features_videomae)

# Apply K-Means Clustering
num_clusters = 4
kmeans_resnet = KMeans(n_clusters=num_clusters, random_state=SEED)
kmeans_videomae = KMeans(n_clusters=num_clusters, random_state=SEED)

clusters_resnet = kmeans_resnet.fit_predict(features_2d_resnet)
clusters_videomae = kmeans_videomae.fit_predict(features_2d_videomae)

# Plot ResNet34 vs VideoMAE with annotations
plt.figure(figsize=(10, 6))
# Plot points first
plt.scatter(features_2d_resnet[:, 0], features_2d_resnet[:, 1], c='blue', label="3D-ResNet34")
plt.scatter(features_2d_videomae[:, 0], features_2d_videomae[:, 1], c='red', label="VideoMAE")

# Annotate ResNet points
for i, filename in enumerate(filenames):
    # Extract tomotarget ID from filename
    tomotarget_id = filename.split('tomotarget')[-1].split('.mrc')[0]
    plt.annotate(tomotarget_id, 
                (features_2d_resnet[i, 0], features_2d_resnet[i, 1]),
                textcoords="offset points",
                xytext=(0,5),
                ha='center',
                fontsize=8)
    
# Annotate Videomae points
for i, filename in enumerate(filenames):
    # Extract tomotarget ID from filename
    tomotarget_id = filename.split('tomotarget')[-1].split('.mrc')[0]
    plt.annotate(tomotarget_id, 
                (features_2d_videomae[i, 0], features_2d_videomae[i, 1]),
                textcoords="offset points",
                xytext=(0,5),
                ha='center',
                fontsize=8)
plt.legend()
plt.title("Comparison of t-SNE Features: 3D-ResNet34 vs. VideoMAE")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.tight_layout()
plt.show()

# add annotations to the K-Means clustering plots

# For ResNet34
plt.figure(figsize=(10, 6))
plt.scatter(features_2d_resnet[:, 0], features_2d_resnet[:, 1], c=clusters_resnet, cmap='viridis')
# Annotate points
for i, filename in enumerate(filenames):
    tomotarget_id = filename.split('tomotarget')[-1].split('.mrc')[0]
    plt.annotate(tomotarget_id, 
                (features_2d_resnet[i, 0], features_2d_resnet[i, 1]),
                textcoords="offset points",
                xytext=(0,5),
                ha='center',
                fontsize=8)
plt.title("K-Means Clustering on 3D-ResNet34 Features")
plt.tight_layout()
plt.show()

# For VideoMAE
plt.figure(figsize=(10, 6))
plt.scatter(features_2d_videomae[:, 0], features_2d_videomae[:, 1], c=clusters_videomae, cmap='coolwarm')
# Annotate points
for i, filename in enumerate(filenames):
    tomotarget_id = filename.split('tomotarget')[-1].split('.mrc')[0]
    plt.annotate(tomotarget_id, 
                (features_2d_videomae[i, 0], features_2d_videomae[i, 1]),
                textcoords="offset points",
                xytext=(0,5),
                ha='center',
                fontsize=8)
plt.title("K-Means Clustering on VideoMAE Features")
plt.tight_layout()
plt.show()

# Compute Silhouette Scores
silhouette_resnet = silhouette_score(features_2d_resnet, clusters_resnet)
silhouette_videomae = silhouette_score(features_2d_videomae, clusters_videomae)

# Compute Davies-Bouldin Index
dbi_resnet = davies_bouldin_score(features_2d_resnet, clusters_resnet)
dbi_videomae = davies_bouldin_score(features_2d_videomae, clusters_videomae)

# Compute Cosine Similarity Between Feature Sets
features_videomae_pooled = features_videomae[:, :512]  # Take first 512 dimensions
cosine_sim = np.mean(cosine_similarity(features_resnet, features_videomae_pooled))

# Report
print("--- Quantitative Analysis Report ---")
print(f"Silhouette Score (Higher is better):")
print(f"  3D-ResNet34: {silhouette_resnet:.4f}")
print(f"  VideoMAE: {silhouette_videomae:.4f}")
print()
print(f"Davies-Bouldin Index (Lower is better):")
print(f"  3D-ResNet34: {dbi_resnet:.4f}")
print(f"  VideoMAE: {dbi_videomae:.4f}")
print()
print(f"Cosine Similarity Between Feature Sets (Higher means more similar features): {cosine_sim:.4f}")
print()

# Interpretation & Implications
if silhouette_resnet > silhouette_videomae:
    print("3D-ResNet34 has better-defined clusters, suggesting more separable features.")
else:
    print("VideoMAE has better-defined clusters, meaning its features are more distinct.")

if dbi_resnet < dbi_videomae:
    print("3D-ResNet34 features are more compact and well-separated.")
else:
    print("VideoMAE features are more compact and well-separated.")

if cosine_sim > 0.5:
    print("There is a significant similarity between the feature spaces of both models.")
else:
    print("The models are learning very different feature representations.")