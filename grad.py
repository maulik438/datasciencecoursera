import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import timm

# -------------------------------
# 1️⃣ Load Model (EfficientNet)
# -------------------------------
model_name = "efficientnet_b0"  # you can use b1, b2, etc.
model = timm.create_model(model_name, pretrained=True)
model.eval()

# -------------------------------
# 2️⃣ Load and Preprocess Image
# -------------------------------
img_path = "your_image.jpg"  # replace with your image
image = Image.open(img_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
input_tensor = transform(image).unsqueeze(0)  # shape [1,3,224,224]

# -------------------------------
# 3️⃣ Define GradCAM Hook
# -------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output)
        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        grad_cam_map = (weights * self.activations).sum(dim=1, keepdim=True)
        grad_cam_map = F.relu(grad_cam_map)
        grad_cam_map = F.interpolate(grad_cam_map, size=(224, 224), mode='bilinear', align_corners=False)
        grad_cam_map = grad_cam_map.squeeze().cpu().numpy()

        grad_cam_map -= grad_cam_map.min()
        grad_cam_map /= grad_cam_map.max()
        return grad_cam_map

# -------------------------------
# 4️⃣ Pick Target Layer for EfficientNet
# -------------------------------
target_layer = model.conv_head  # works for timm EfficientNet

grad_cam = GradCAM(model, target_layer)
cam = grad_cam.generate(input_tensor)

# -------------------------------
# 5️⃣ Overlay GradCAM on Image
# -------------------------------
img_cv = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
img_cv = cv2.resize(img_cv, (224, 224))

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = 0.4 * heatmap + 0.6 * img_cv

plt.figure(figsize=(10,5))
plt.subplot(1,3,1)
plt.title('Original')
plt.imshow(img_cv)
plt.axis('off')

plt.subplot(1,3,2)
plt.title('GradCAM Heatmap')
plt.imshow(cam, cmap='jet')
plt.axis('off')

plt.subplot(1,3,3)
plt.title('Overlay')
plt.imshow(np.uint8(overlay))
plt.axis('off')
plt.show()
