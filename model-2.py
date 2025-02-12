import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm  # Import Xception model
from tqdm import tqdm

# Image Transformations (128x128 resolution)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

# Dataset Loader for DFDC Frames
class DFDCFrameDataset(Dataset):
    def __init__(self, root_dir, num_frames=5, split="TEST"):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.data = []

        for label, category in enumerate(["REAL", "FAKE"]):  
            category_path = os.path.join(root_dir, category, split)
            if not os.path.exists(category_path):
                continue

            for video_id in os.listdir(category_path):
                video_path = os.path.join(category_path, video_id)
                if os.path.isdir(video_path):  
                    frame_files = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))])
                    if len(frame_files) >= num_frames:
                        self.data.append((frame_files, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frame_files, label = self.data[idx]
        indices = torch.linspace(0, len(frame_files) - 1, self.num_frames).long()  
        frames = [transform(Image.open(frame_files[i]).convert("RGB")) for i in indices]
        return torch.stack(frames), torch.tensor(label, dtype=torch.float32)

# Frame-Level Feature Extractor (Xception)
class XceptionFeatureExtractor(nn.Module):
    def __init__(self, embed_dim=2048):  
        super().__init__()
        self.xception = timm.create_model('xception', pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.xception.children())[:-1])  # Remove classifier
        self.fc = nn.Linear(embed_dim, 768)  # Reduce dimension

    def forward(self, x):
        x = self.feature_extractor(x)  
        x = x.view(x.size(0), -1)  
        return self.fc(x)  

# Transformer Encoder (Processes Sequence of Frame Features)
class TemporalTransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, num_layers=2, num_heads=8):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.encoder(x)  

# Cross-Attention Layer
class CrossAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x, reference):
        return self.attn(x, reference, reference)[0]  

# Feed-Forward Network (FFN) for Classification
class FFN(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  
        )

    def forward(self, x):
        return self.fc(x)

# Full Model: Xception + Transformer Encoder + Cross-Attention + FFN
class ForgeryDetector(nn.Module):
    def __init__(self, embed_dim=768, num_layers=2, num_heads=8):
        super().__init__()
        self.frame_extractor = XceptionFeatureExtractor(embed_dim=2048)  
        self.temporal_encoder = TemporalTransformerEncoder(embed_dim, num_layers, num_heads)
        self.cross_attention = CrossAttention(embed_dim, num_heads)
        self.ffn = FFN(embed_dim)

    def forward(self, frames, reference):
        batch_size, num_frames, c, h, w = frames.shape  
        
        # Extract frame-level features
        frame_features = []
        for t in range(num_frames):
            feature = self.frame_extractor(frames[:, t, :, :, :])  
            frame_features.append(feature.unsqueeze(1))  

        frame_features = torch.cat(frame_features, dim=1)  

        # Process sequence with Transformer Encoder
        group_features = self.temporal_encoder(frame_features)  

        # Apply Cross-Attention (Compare with Reference)
        reference = reference.unsqueeze(1)  
        attended_features = self.cross_attention(group_features, reference)  

        # Pooling over time (average across frames)
        final_representation = attended_features.mean(dim=1)  

        # Binary Classification Output
        return self.ffn(final_representation)  

# Load Untrained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ForgeryDetector().to(device)
model.eval()

# Load the test dataset
dataset_path = "./DFDC"  # Update if needed
test_dataset = DFDCFrameDataset(dataset_path, num_frames=64, split="TEST")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # 1 image per inference

# Initialize Counters
total_images = 0
real_count = 0
fake_count = 0
correct_predictions = 0  # Tracks correct classifications

print("\n🔍 Running Untrained Model Inference on Test Data...\n")
with torch.no_grad():
    for frames, label in tqdm(test_loader, desc="Processing"):
        frames = frames.to(device)
        reference_embedding = torch.randn(frames.shape[0], 768).to(device)  

        output = model(frames, reference_embedding).squeeze().item()  
        prediction = 1 if output >= 0.5 else 0  

        # Count occurrences
        total_images += 1
        if prediction == 1:
            fake_count += 1
        else:
            real_count += 1

        # Track correct predictions
        if prediction == label.item():
            correct_predictions += 1

        print(f"📌 Image {total_images}: Predicted = {'FAKE' if prediction == 1 else 'REAL'}, "
              f"Actual = {'FAKE' if label.item() == 1 else 'REAL'}, "
              f"{'✅ Correct' if prediction == label.item() else '❌ Incorrect'}")

# Compute Statistics
real_percentage = (real_count / total_images) * 100
fake_percentage = (fake_count / total_images) * 100
accuracy = (correct_predictions / total_images) * 100  # Compute accuracy

# Display Final Results
print("\n📊 **Inference Statistics (Before Training)**")
print(f"🔹 Total Images Processed: {total_images}")
print(f"🟢 Real Predictions: {real_count} ({real_percentage:.2f}%)")
print(f"🔴 Fake Predictions: {fake_count} ({fake_percentage:.2f}%)")
print(f"✅ Correct Predictions: {correct_predictions} ({accuracy:.2f}%) Accuracy")
