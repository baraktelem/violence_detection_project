# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms

import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from model import Generator  # ודא שקובץ model.py מהמאגר נמצא באותה תיקייה

# הגדרת המכשיר (GPU אם זמין, אחרת CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# טעינת המודל והמשקולות
model = Generator().to(device)
model.load_state_dict(torch.load('SRGAN_x4-SRGAN_ImageNet.pth.tar', map_location=device)['state_dict'])
model.eval()

# הורדת תמונה אקראית מהאינטרנט
url = 'https://i.sstatic.net/etJFB.jpg'  # החלף ב-URL של תמונה לבחירתך
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert('RGB')

# הצגת התמונה המקורית
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img)
plt.axis('off')

# עיבוד התמונה והעברת המודל
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.unsqueeze(0))
])
img_lr = transform(img).to(device)

with torch.no_grad():
    img_sr = model(img_lr)

img_sr = img_sr.squeeze(0).cpu().clamp(0, 1)

# הצגת התמונה המשוחזרת
plt.subplot(1, 2, 2)
plt.title('Super-Resolved Image')
plt.imshow(transforms.ToPILImage()(img_sr))
plt.axis('off')

plt.show()