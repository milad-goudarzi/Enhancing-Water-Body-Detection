import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

IMAGE_SIZE = (400, 400)
IMAGE_PATH = '/kaggle/input/glacial-lake-dataset/glacial-lake-dataset/images'
MASK_PATH = '/kaggle/input/glacial-lake-dataset/glacial-lake-dataset/masks'


def load_image_tf(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, IMAGE_SIZE, method='nearest')
    mask = tf.cast(mask > 127, tf.float32)
    return img, mask

def load_data(image_path, mask_path):
    image_files = sorted(os.listdir(image_path))
    mask_files = sorted(os.listdir(mask_path))

    X, Y = [], []
    for img_file, mask_file in zip(image_files, mask_files):
        img_full_path = os.path.join(image_path, img_file)
        mask_full_path = os.path.join(mask_path, mask_file)

        img, mask = load_image_tf(img_full_path, mask_full_path)
        X.append(img.numpy())   # Convert from EagerTensor to NumPy array
        Y.append(mask.numpy())

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# Load and split
X, Y = load_data(IMAGE_PATH, MASK_PATH)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

def get_tf_model():
    inputs = tf.keras.Input(shape=(400, 400, 3))
    x = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D()(x)
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, outputs)

model = get_tf_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start_train = time.time()
history = model.fit(X_train, Y_train, epochs=15, batch_size=8, validation_data=(X_val, Y_val), verbose=1)
train_time = time.time() - start_train

df = pd.DataFrame(history.history)
filename = "tensorflow_training_history.csv"
df.to_csv(filename, index=False)


start_infer = time.time()
tf_preds = model.predict(X_val)
infer_time = time.time() - start_infer

preds_bin = (tf_preds > 0.5).astype(np.uint8)
y_true_flat = Y_val.flatten()
y_pred_flat = preds_bin.flatten()

print("TensorFlow")
print("Training Time:", round(train_time, 2), "s")
print("Inference Time:", round(infer_time, 2), "s")
print("Precision:", precision_score(y_true_flat, y_pred_flat))
print("Recall:", recall_score(y_true_flat, y_pred_flat))
print("F1 Score:", f1_score(y_true_flat, y_pred_flat))
print("IoU:", jaccard_score(y_true_flat, y_pred_flat))

class LakeDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images.transpose(0, 3, 1, 2)  # HWC to CHW
        self.masks = masks.transpose(0, 3, 1, 2)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.tensor(self.images[idx]), torch.tensor(self.masks[idx])

train_ds = LakeDataset(X_train, Y_train)
val_ds = LakeDataset(X_val, Y_val)
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=8)

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.max_pool2d(x1, 2)
        x3 = F.relu(self.conv2(x2))
        x4 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        return torch.sigmoid(self.conv3(x4))


model = SimpleUNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

metrics = []  # Store each epoch's metrics

start_train = time.time()
for epoch in range(15):
    model.train()
    total_train_loss = 0
    num_train_batches = 0

    for xb, yb in train_dl:
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        num_train_batches += 1

    avg_train_loss = total_train_loss / num_train_batches

    # === Validation Loss ===
    model.eval()
    total_val_loss = 0
    num_val_batches = 0

    with torch.no_grad():
        for xb, yb in val_dl:
            preds = model(xb)
            val_loss = loss_fn(preds, yb)
            total_val_loss += val_loss.item()
            num_val_batches += 1

    avg_val_loss = total_val_loss / num_val_batches

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save metrics
    metrics.append({
        'epoch': epoch + 1,
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss
    })

train_time = time.time() - start_train

df = pd.DataFrame(metrics)
df.to_csv("pytorch_training_history.csv", index=False)


start_infer = time.time()
model.eval()
with torch.no_grad():
    all_preds, all_labels = [], []
    for xb, yb in val_dl:
        out = model(xb)
        all_preds.append(out)
        all_labels.append(yb)
        loss = loss_fn(out, yb)
infer_time = time.time() - start_infer


preds = torch.cat(all_preds).numpy()
labels = torch.cat(all_labels).numpy()
preds_bin = (preds > 0.5).astype(np.uint8)
y_true_flat = labels.flatten()
y_pred_flat = preds_bin.flatten()

print("PyTorch")
print("Training Time:", round(train_time, 2), "s")
print("Inference Time:", round(infer_time, 2), "s")
print("Precision:", precision_score(y_true_flat, y_pred_flat))
print("Recall:", recall_score(y_true_flat, y_pred_flat))
print("F1 Score:", f1_score(y_true_flat, y_pred_flat))
print("IoU:", jaccard_score(y_true_flat, y_pred_flat))

pt_history = pd.read_csv('pytorch_training_history.csv')
tf_history = pd.read_csv('tensorflow_training_history.csv')

plt.figure(figsize=(10, 6))
plt.plot(pt_history["epoch"], pt_history["val_loss"], marker='o', label='PyTorch val_loss')
plt.plot(range(1, 16), tf_history["val_loss"], marker='o', label='TensorFlow val_loss')

plt.title("Validation Loss Comparison: PyTorch vs TensorFlow")
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def visualize_comparison(images, masks, tf_preds, torch_preds, num_samples=5):
    plt.figure(figsize=(16, num_samples * 3))
    for i in range(num_samples):
        plt.subplot(num_samples, 4, i * 4 + 1)
        plt.imshow(images[i])
        plt.title("Image")
        plt.axis("off")

        plt.subplot(num_samples, 4, i * 4 + 2)
        plt.imshow(masks[i].squeeze(), cmap='gray')
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(num_samples, 4, i * 4 + 3)
        plt.imshow(tf_preds[i].squeeze(), cmap='gray')
        plt.title("TensorFlow")
        plt.axis("off")

        plt.subplot(num_samples, 4, i * 4 + 4)
        plt.imshow(torch_preds[i].squeeze(), cmap='gray')
        plt.title("PyTorch")
        plt.axis("off")

    plt.suptitle("Segmentation Comparison: Image | Ground Truth | TF | PyTorch", fontsize=16)
    plt.tight_layout()
    plt.show()



torch_preds_np = torch.cat(all_preds).numpy()
torch_preds_np = (torch_preds_np > 0.5).astype(np.uint8)

tf_preds_np = (tf_preds > 0.5).astype(np.uint8)

true_masks = np.concatenate([y for _, y in val_ds], axis=0)

visualize_comparison(X_val, true_masks, tf_preds_np, torch_preds_np)

