# Quick visualization script (verify_samples.py)
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# Load annotations
df = pd.read_csv("data/train/annotations/annotations.csv")
sample = df.iloc[0]

# Display sample
img = cv2.imread(f"data/train/images/{sample['image_path']}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8,8))
plt.imshow(img_rgb)
plt.scatter(sample['x'], sample['y'], c='cyan', s=50, marker='x')
plt.title(f"Target at ({sample['x']:.1f}, {sample['y']:.1f})")
plt.show()
