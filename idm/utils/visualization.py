from __future__ import absolute_import

import os
import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from scipy.ndimage import zoom
    SCIPY_NDIMAGE_AVAILABLE = True
except ImportError:
    SCIPY_NDIMAGE_AVAILABLE = False
    print("Warning: scipy.ndimage not available. Some resizing operations may fail.")

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False
    print("Warning: scikit-learn not available. t-SNE visualization disabled.")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn not available. UMAP visualization disabled.")


def extract_features_for_visualization(model, data_loader, device='cuda'):
    """
    Extract features and labels from a data loader for visualization.
    
    Args:
        model: PyTorch model (should be in eval mode)
        data_loader: DataLoader providing (imgs, fnames, pids, camids, ...)
        device: Device to run on
    
    Returns:
        features: numpy array of shape [N, D] where N is number of samples, D is feature dimension
        labels: numpy array of shape [N] with person IDs
        fnames: list of filenames
    """
    model.eval()
    features_list = []
    labels_list = []
    fnames_list = []
    
    with torch.no_grad():
        for batch in data_loader:
            imgs = batch[0].to(device)
            fnames = batch[1]
            pids = batch[2]
            
            # Extract features
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                # If model returns (logits, features), take features
                outputs = outputs[1] if len(outputs) > 1 else outputs[0]
            
            # Normalize features
            outputs = F.normalize(outputs, p=2, dim=1)
            
            features_list.append(outputs.cpu().numpy())
            labels_list.extend(pids.numpy())
            fnames_list.extend(fnames)
    
    features = np.vstack(features_list)
    labels = np.array(labels_list)
    
    return features, labels, fnames_list


def visualize_tsne(features, labels, save_path, title='t-SNE Visualization', 
                   n_components=2, perplexity=30, n_iter=1000, random_state=42):
    """
    Perform t-SNE dimensionality reduction and create scatter plot.
    
    Args:
        features: numpy array of shape [N, D] feature vectors
        labels: numpy array of shape [N] with class labels (person IDs)
        save_path: Path to save the plot
        title: Plot title
        n_components: Number of dimensions for t-SNE (2 or 3)
        perplexity: t-SNE perplexity parameter
        n_iter: Number of iterations
        random_state: Random seed
    """
    if not TSNE_AVAILABLE:
        raise ImportError("scikit-learn is required for t-SNE visualization. Install with: pip install scikit-learn")
    
    print(f"Running t-SNE on {len(features)} samples with {features.shape[1]} dimensions...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                n_iter=n_iter, random_state=random_state, verbose=1)
    features_2d = tsne.fit_transform(features)
    
    # Create scatter plot
    plt.figure(figsize=(12, 10))
    
    # Get unique labels and assign colors
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, colors))
    
    # Plot each class with different color
    for label in unique_labels:
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[label_to_color[label]], label=f'ID {label}', 
                   alpha=0.6, s=20)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"t-SNE plot saved to: {save_path}")
    plt.close()


def visualize_umap(features, labels, save_path, title='UMAP Visualization',
                   n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Perform UMAP dimensionality reduction and create scatter plot.
    
    Args:
        features: numpy array of shape [N, D] feature vectors
        labels: numpy array of shape [N] with class labels (person IDs)
        save_path: Path to save the plot
        title: Plot title
        n_components: Number of dimensions for UMAP (2 or 3)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        random_state: Random seed
    """
    if not UMAP_AVAILABLE:
        raise ImportError("umap-learn is required for UMAP visualization. Install with: pip install umap-learn")
    
    print(f"Running UMAP on {len(features)} samples with {features.shape[1]} dimensions...")
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                        min_dist=min_dist, random_state=random_state, verbose=True)
    features_2d = reducer.fit_transform(features)
    
    # Create scatter plot
    plt.figure(figsize=(12, 10))
    
    # Get unique labels and assign colors
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, colors))
    
    # Plot each class with different color
    for label in unique_labels:
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=[label_to_color[label]], label=f'ID {label}',
                   alpha=0.6, s=20)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"UMAP plot saved to: {save_path}")
    plt.close()


def generate_gradcam(model, img_tensor, target_layer, device='cuda', 
                     class_idx=None, reshape_transform=None):
    """
    Generate Grad-CAM heatmap for an image.
    
    Args:
        model: PyTorch model
        img_tensor: Input image tensor [1, C, H, W]
        target_layer: Target layer to compute gradients for (e.g., model.layer4)
        device: Device to run on
        class_idx: Class index to generate CAM for (None = use highest logit)
        reshape_transform: Function to reshape activations if needed
    
    Returns:
        cam: Grad-CAM heatmap as numpy array [H, W]
    """
    model.eval()
    img_tensor = img_tensor.to(device)
    
    # Register forward hook to capture activations
    activations = []
    gradients = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_backward_hook(backward_hook)
    
    try:
        # Forward pass
        output = model(img_tensor)
        if isinstance(output, tuple):
            output = output[0]  # Take logits
        
        # Backward pass
        if class_idx is None:
            # Use highest scoring class
            class_idx = output.argmax(dim=1)
        
        model.zero_grad()
        output[0, class_idx].backward()
        
        # Get gradients and activations
        if len(gradients) == 0 or len(activations) == 0:
            raise ValueError("Failed to capture gradients or activations")
        
        grads = gradients[0]
        acts = activations[0]
        
        # Reshape if needed
        if reshape_transform is not None:
            grads = reshape_transform(grads)
            acts = reshape_transform(acts)
        
        # Compute weights (global average pooling of gradients)
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * acts, dim=1, keepdim=True)  # [B, 1, H, W]
        cam = F.relu(cam)  # Apply ReLU
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    finally:
        handle_forward.remove()
        handle_backward.remove()


def visualize_gradcam(model, img_path, save_path, target_layer, device='cuda',
                      class_idx=None, alpha=0.4, resize_to=(256, 128)):
    """
    Generate and save Grad-CAM visualization for an image.
    
    Args:
        model: PyTorch model
        img_path: Path to input image
        save_path: Path to save visualization
        target_layer: Target layer for Grad-CAM (e.g., model.layer4)
        device: Device to run on
        class_idx: Class index for CAM (None = use highest logit)
        alpha: Transparency for heatmap overlay
        resize_to: Resize image to this size (H, W)
    """
    from torchvision import transforms
    
    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    original_size = img.size
    
    # Resize if needed
    if resize_to:
        img = img.resize((resize_to[1], resize_to[0]), Image.LANCZOS)
    
    # Preprocess for model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]
    
    # Generate Grad-CAM
    cam = generate_gradcam(model, img_tensor, target_layer, device, class_idx)
    
    # Resize CAM to original image size if needed
    if resize_to:
        if not SCIPY_NDIMAGE_AVAILABLE:
            raise ImportError("scipy.ndimage is required for Grad-CAM visualization. Install with: pip install scipy")
        cam = zoom(cam, (original_size[1] / resize_to[0], original_size[0] / resize_to[1]), order=1)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(Image.open(img_path))
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # Heatmap
    im = axes[1].imshow(cam, cmap='jet', interpolation='bilinear')
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # Overlay
    img_overlay = np.array(Image.open(img_path))
    if img_overlay.shape[:2] != cam.shape:
        # Resize image to match CAM
        if not SCIPY_NDIMAGE_AVAILABLE:
            raise ImportError("scipy.ndimage is required for Grad-CAM visualization. Install with: pip install scipy")
        zoom_factors = (cam.shape[0] / img_overlay.shape[0], 
                       cam.shape[1] / img_overlay.shape[1], 1)
        img_overlay = zoom(img_overlay, zoom_factors, order=1)
        img_overlay = np.clip(img_overlay, 0, 255).astype(np.uint8)
    
    # Apply colormap to CAM
    cam_colored = plt.cm.jet(cam)[:, :, :3]  # Remove alpha channel
    cam_colored = (cam_colored * 255).astype(np.uint8)
    
    # Blend
    overlay = (alpha * cam_colored + (1 - alpha) * img_overlay).astype(np.uint8)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=12)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Grad-CAM visualization saved to: {save_path}")
    plt.close()


def visualize_multiple_gradcam(model, img_paths, save_path, target_layer, 
                               device='cuda', class_idx=None, alpha=0.4,
                               n_cols=4, resize_to=(256, 128)):
    """
    Generate Grad-CAM visualizations for multiple images in a grid.
    
    Args:
        model: PyTorch model
        img_paths: List of image paths
        save_path: Path to save visualization
        target_layer: Target layer for Grad-CAM
        device: Device to run on
        class_idx: Class index for CAM (None = use highest logit)
        alpha: Transparency for heatmap overlay
        n_cols: Number of columns in grid
        resize_to: Resize images to this size
    """
    n_imgs = len(img_paths)
    n_rows = (n_imgs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, img_path in enumerate(img_paths):
        ax = axes[idx]
        
        try:
            # Load and preprocess
            img = Image.open(img_path).convert('RGB')
            if resize_to:
                img = img.resize((resize_to[1], resize_to[0]), Image.LANCZOS)
            
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            img_tensor = transform(img).unsqueeze(0)
            
            # Generate Grad-CAM
            cam = generate_gradcam(model, img_tensor, target_layer, device, class_idx)
            
            # Resize CAM to match image
            if not SCIPY_NDIMAGE_AVAILABLE:
                raise ImportError("scipy.ndimage is required for Grad-CAM visualization. Install with: pip install scipy")
            
            cam = zoom(cam, (img.size[1] / resize_to[0], img.size[0] / resize_to[1]), order=1)
            
            # Create overlay
            img_array = np.array(img)
            if img_array.shape[:2] != cam.shape:
                if not SCIPY_NDIMAGE_AVAILABLE:
                    raise ImportError("scipy.ndimage is required for Grad-CAM visualization. Install with: pip install scipy")
                
                zoom_factors = (cam.shape[0] / img_array.shape[0],
                              cam.shape[1] / img_array.shape[1], 1)
                img_array = zoom(img_array, zoom_factors, order=1)
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
            cam_colored = plt.cm.jet(cam)[:, :, :3]
            cam_colored = (cam_colored * 255).astype(np.uint8)
            overlay = (alpha * cam_colored + (1 - alpha) * img_array).astype(np.uint8)
            
            ax.imshow(overlay)
            ax.set_title(osp.basename(img_path), fontsize=10)
            ax.axis('off')
        
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
            ax.axis('off')
    
    # Hide unused subplots
    for idx in range(n_imgs, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Multi-image Grad-CAM visualization saved to: {save_path}")
    plt.close()


