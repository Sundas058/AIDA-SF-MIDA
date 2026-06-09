from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.append(".")
from idm import datasets
from idm import models
from idm.models.dsbn import convert_dsbn
from idm.models.idm_dsbn import convert_dsbn_idm, convert_bn_idm
from idm.utils.data import transforms as T
from idm.utils.data.preprocessor import Preprocessor
from idm.utils.serialization import load_checkpoint, copy_state_dict
from idm.utils.visualization import (
    extract_features_for_visualization,
    visualize_tsne,
    visualize_umap,
    visualize_gradcam,
    visualize_multiple_gradcam
)


def get_data(name, data_dir, height, width, batch_size, workers, subset_size=None):
    """Get data loader, optionally limiting to a subset."""
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    
    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])
    
    # Optionally limit dataset size for faster visualization
    data_list = list(set(dataset.query) | set(dataset.gallery))
    if subset_size and subset_size < len(data_list):
        import random
        random.seed(42)
        data_list = random.sample(data_list, subset_size)
    
    test_loader = DataLoader(
        Preprocessor(data_list, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    
    return dataset, test_loader


def filter_layers(stage):
    """Filter layer names for IDM."""
    layer_names = ['conv', 'layer1', 'layer2', 'layer3', 'layer4', 'feat_bn']
    ori_bn_names = []
    idm_bn_names = []
    for i in range(len(layer_names)):
        if i < stage+1:
            ori_bn_names.append(layer_names[i])
        else:
            idm_bn_names.append(layer_names[i])
    return idm_bn_names


def get_target_layer(model, layer_name='layer4'):
    """
    Get target layer for Grad-CAM.
    
    Args:
        model: PyTorch model
        layer_name: Name of layer (e.g., 'layer4', 'layer3')
    
    Returns:
        Target layer module
    """
    if hasattr(model, 'module'):
        model = model.module
    
    if hasattr(model, layer_name):
        return getattr(model, layer_name)
    elif hasattr(model, 'base'):
        # For IDM models
        if hasattr(model.base, layer_name):
            return getattr(model.base, layer_name)
    
    raise ValueError(f"Layer '{layer_name}' not found in model. Available layers: {list(model.named_modules())}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize features using t-SNE/UMAP and generate Grad-CAM visualizations"
    )
    
    # Data
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        choices=datasets.names())
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to dataset root')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--subset-size', type=int, default=None,
                        help='Limit number of samples for faster visualization (optional)')
    
    # Model
    parser.add_argument('--resume', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--stage', type=int, default=0,
                        help='IDM stage (for dsbn-idm)')
    parser.add_argument('--dsbn', action='store_true',
                        help='Use domain-specific BN')
    parser.add_argument('--dsbn-idm', action='store_true',
                        help='Use domain-specific BN (IDM)')
    parser.add_argument('--test-source', action='store_true',
                        help='Test on source domain (for dsbn)')
    
    # Visualization options
    parser.add_argument('--visualization-type', type=str, required=True,
                        choices=['tsne', 'umap', 'gradcam', 'all'],
                        help='Type of visualization to generate')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                        help='Directory to save visualizations')
    
    # t-SNE/UMAP options
    parser.add_argument('--tsne-perplexity', type=int, default=30,
                        help='t-SNE perplexity parameter')
    parser.add_argument('--tsne-iter', type=int, default=1000,
                        help='t-SNE iterations')
    parser.add_argument('--umap-neighbors', type=int, default=15,
                        help='UMAP n_neighbors parameter')
    parser.add_argument('--umap-mindist', type=float, default=0.1,
                        help='UMAP min_dist parameter')
    
    # Grad-CAM options
    parser.add_argument('--gradcam-layer', type=str, default='layer4',
                        help='Target layer for Grad-CAM (e.g., layer4, layer3)')
    parser.add_argument('--gradcam-images', type=str, nargs='+', default=None,
                        help='Paths to images for Grad-CAM (if not provided, samples from dataset)')
    parser.add_argument('--gradcam-num-images', type=int, default=8,
                        help='Number of images to visualize with Grad-CAM (if --gradcam-images not provided)')
    parser.add_argument('--gradcam-class-idx', type=int, default=None,
                        help='Class index for Grad-CAM (None = use highest logit)')
    
    args = parser.parse_args()
    
    # Create model
    model = models.create(args.arch, pretrained=False, num_features=args.features,
                          dropout=args.dropout, num_classes=0)
    
    if args.dsbn:
        convert_dsbn(model)
    if args.dsbn_idm:
        idm_bn_names = filter_layers(args.stage)
        convert_dsbn_idm(model, idm_bn_names, idm=False)
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.resume)
    copy_state_dict(checkpoint['state_dict'], model, strip='module.')
    
    if args.dsbn:
        convert_bn(model, use_target=(not args.test_source))
    if args.dsbn_idm:
        convert_bn_idm(model, use_target=(not args.test_source))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model = nn.DataParallel(model)
    model.eval()
    
    print(f"Model loaded from: {args.resume}")
    print(f"Visualization type: {args.visualization_type}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # t-SNE or UMAP visualization
    if args.visualization_type in ['tsne', 'umap', 'all']:
        print("\n" + "="*60)
        print("Extracting features for dimensionality reduction...")
        print("="*60)
        
        dataset, test_loader = get_data(args.dataset, args.data_dir, args.height,
                                       args.width, args.batch_size, args.workers,
                                       subset_size=args.subset_size)
        
        features, labels, fnames = extract_features_for_visualization(
            model, test_loader, device=device
        )
        
        print(f"Extracted features: {features.shape}")
        print(f"Number of unique identities: {len(set(labels))}")
        
        if args.visualization_type in ['tsne', 'all']:
            print("\nGenerating t-SNE visualization...")
            tsne_path = osp.join(args.output_dir, f'tsne_{args.dataset}.png')
            visualize_tsne(
                features, labels, tsne_path,
                title=f't-SNE Visualization - {args.dataset}',
                perplexity=args.tsne_perplexity,
                n_iter=args.tsne_iter
            )
        
        if args.visualization_type in ['umap', 'all']:
            print("\nGenerating UMAP visualization...")
            umap_path = osp.join(args.output_dir, f'umap_{args.dataset}.png')
            visualize_umap(
                features, labels, umap_path,
                title=f'UMAP Visualization - {args.dataset}',
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_mindist
            )
    
    # Grad-CAM visualization
    if args.visualization_type in ['gradcam', 'all']:
        print("\n" + "="*60)
        print("Generating Grad-CAM visualizations...")
        print("="*60)
        
        # Get target layer
        target_layer = get_target_layer(model.module, args.gradcam_layer)
        print(f"Using layer: {args.gradcam_layer}")
        
        # Get images
        if args.gradcam_images:
            img_paths = args.gradcam_images
        else:
            # Sample images from dataset
            dataset, test_loader = get_data(args.dataset, args.data_dir, args.height,
                                          args.width, args.batch_size, args.workers)
            
            # Get some sample images
            img_paths = []
            for batch in test_loader:
                fnames = batch[1]
                root = osp.join(args.data_dir, args.dataset)
                for fname in fnames:
                    img_path = osp.join(root, 'bounding_box_train', fname)
                    if not osp.exists(img_path):
                        img_path = osp.join(root, 'bounding_box_test', fname)
                    if osp.exists(img_path):
                        img_paths.append(img_path)
                    if len(img_paths) >= args.gradcam_num_images:
                        break
                if len(img_paths) >= args.gradcam_num_images:
                    break
        
        print(f"Visualizing {len(img_paths)} images with Grad-CAM...")
        
        if len(img_paths) == 1:
            # Single image
            gradcam_path = osp.join(args.output_dir, f'gradcam_{osp.basename(img_paths[0])}.png')
            visualize_gradcam(
                model, img_paths[0], gradcam_path, target_layer,
                device=device, class_idx=args.gradcam_class_idx
            )
        else:
            # Multiple images in grid
            gradcam_path = osp.join(args.output_dir, f'gradcam_{args.dataset}_grid.png')
            visualize_multiple_gradcam(
                model, img_paths[:args.gradcam_num_images], gradcam_path, target_layer,
                device=device, class_idx=args.gradcam_class_idx
            )
    
    print("\n" + "="*60)
    print("Visualization complete!")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    from idm.models.dsbn import convert_bn
    main()

