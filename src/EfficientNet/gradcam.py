import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import tifffile
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.io as io
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from data_loader import get_valid_transform
from utils import FourChannelTiffDataset
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description='Grad-CAM batch runner for 4-channel EfficientNet')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Root directory of test data (mutants/ and wild_type/)')
    parser.add_argument('--output_dir', type=str, default='./gradcam_outputs',
                        help='Directory to save Grad-CAM images')
    parser.add_argument('--target_layer', type=str, default='_conv_head',
                        help='Suffix of layer name to hook for Grad-CAM')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for data loader')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Max number of samples to process')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Computation device')
    return parser.parse_args()

class GradCAM:
    def __init__(self, model, target_layer_suffix):
        self.model = model
        self.gradients = None
        self.activations = None
        for name, module in model.named_modules():
            if name.endswith(target_layer_suffix):
                module.register_forward_hook(self._forward_hook)
                module.register_full_backward_hook(self._backward_hook)
                break
        else:
            raise ValueError(f"No layer ending with '{target_layer_suffix}' found.")

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_cam(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        class_idx = output.argmax(dim=1).item()
        output[0, class_idx].backward()
        grads = self.gradients[0]
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = (weights * self.activations[0]).sum(dim=0)
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), class_idx

def overlay_and_save(img_path, cam, save_path):
    """
    Overlay the Grad-CAM on the image and save it.

    Parameters:
    - img_path: Path to the original 4-channel image (TIFF).
    - cam: 2D numpy array with shape [H, W], values in [0, 1].
    - save_path: Path to save the resulting visualization.
    """
    logger.info(f"Processing {img_path}")
    try:
        raw = tifffile.imread(img_path)  # Shape: (H, W, 4)
        if raw.ndim != 3 or raw.shape[-1] != 4:
            raise ValueError(f"Expected 3D array with 4 channels, got shape {raw.shape}")
    except Exception as e:
        logger.error(f"Failed to load {img_path}: {e}")
        return

    # Split channels
    phase = raw[:, :, 0]  # Grayscale background
    dapi  = raw[:, :, 1]  # Blue
    ftsZ  = raw[:, :, 2]  # Green
    seqA  = raw[:, :, 3]  # Red

    # Normalize each channel to [0, 1]
    def normalize(channel):
        if channel.max() == channel.min():
            return np.zeros_like(channel, dtype=np.float32)
        return (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)

    phase = normalize(phase)
    dapi  = normalize(dapi)
    ftsZ  = normalize(ftsZ)
    seqA  = normalize(seqA)

    # Combine RGB channels (fluorescence)
    fluorescence_rgb = np.stack([seqA, ftsZ, dapi], axis=-1)  # [H, W, 3]

    # Convert phase to 3-channel grayscale
    phase_rgb = np.stack([phase] * 3, axis=-1)  # [H, W, 3]

    # Overlay fluorescence on phase
    alpha = 0.6
    rgb = (1 - alpha) * phase_rgb + alpha * fluorescence_rgb
    rgb = np.clip(rgb, 0, 1)

    # Resize CAM to image size
    cam_resized = TF.resize(TF.to_pil_image(torch.tensor(cam)), size=rgb.shape[:2])
    cam_resized = np.array(cam_resized) / 255.0  # Normalize to [0, 1]

    # Convert CAM to color map
    heatmap = plt.cm.jet(cam_resized)[..., :3]  # Ignore alpha channel

    # Overlay CAM on image
    overlay = 0.5 * rgb + 0.5 * heatmap
    overlay = np.clip(overlay, 0, 1)

    # Save the result
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.imsave(save_path, overlay)
    logger.info(f"Saved Grad-CAM overlay to: {save_path}")

def main():
    args = get_args()
    device = args.device
    logger.info(f"Using device: {device}")

    # Dataset & Loader
    transform = get_valid_transform((224, 224))
    dataset = FourChannelTiffDataset(args.data_dir, transform=transform)
    logger.info(f"Classes: {dataset.classes}")
    logger.info(f"Total images: {len(dataset.images)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
    model._conv_stem = torch.nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if any(k.startswith('module.') for k in state):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()

    # GradCAM
    cam_util = GradCAM(model, args.target_layer)
    processed = 0

    for idx, (imgs, labels) in enumerate(loader):
        if processed >= args.num_samples:
            break
        imgs = imgs.to(device)
        cam, pred = cam_util.generate_cam(imgs)
        true_cls = dataset.classes[labels.item()]
        pred_cls = dataset.classes[pred]
        img_path, _ = dataset.images[idx]
        fname = f"{idx}_true_{true_cls}_pred_{pred_cls}.png"
        # Save in class-specific subdirectory
        save_path = os.path.join(args.output_dir, true_cls, fname)
        logger.info(f"Processing image {idx}: True class = {true_cls}, Predicted class = {pred_cls}")
        overlay_and_save(img_path, cam, save_path)
        processed += 1

    logger.info(f"Completed {processed} samples.")

if __name__ == '__main__':
    main()