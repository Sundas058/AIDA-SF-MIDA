from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta
from sklearn.cluster import DBSCAN
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append(".")
from idm import datasets
from idm.datasets.multi_source import MultiSourceDataset
from idm import models
from idm.models.idm_dsbn import convert_dsbn_idm, convert_bn_idm
from idm.models.xbm import XBM
from idm.trainers import Baseline_Trainer, IDM_Trainer
from idm.evaluators import Evaluator, extract_features
from idm.utils.data import IterLoader
from idm.utils.data import transforms as T
from idm.utils.data.sampler import RandomMultipleGallerySampler, MultiSourceSampler
from idm.utils.data.preprocessor import Preprocessor
from idm.utils.logging import Logger
from idm.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from idm.utils.rerank import compute_jaccard_distance


start_epoch = best_mAP = 0

def get_data(name, data_dir, source_names=None):
    """
    Get dataset. If source_names is provided and has length > 1, returns MultiSourceDataset.
    Otherwise returns a single dataset (backward compatible).
    """
    if source_names is not None and len(source_names) > 1:
        # Multi-source mode
        dataset = MultiSourceDataset(source_names, data_dir, verbose=True)
        return dataset
    else:
        # Single-source mode (backward compatible)
        root = osp.join(data_dir, name)
        dataset = datasets.create(name, root)
        return dataset

def get_train_loader(args, dataset, height, width, batch_size, workers,
                    num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
	         T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    
    # Check if this is a multi-source dataset (samples have domain_id)
    is_multi_source = len(train_set) > 0 and len(train_set[0]) >= 4
    
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        # Use MultiSourceSampler for multi-source datasets, otherwise use RandomMultipleGallerySampler
        if is_multi_source:
            sampler = MultiSourceSampler(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    
    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def filter_layers(stage):
    layer_names = ['conv', 'layer1', 'layer2', 'layer3', 'layer4', 'feat_bn']
    ori_bn_names = []
    idm_bn_names = []
    for i in range(len(layer_names)):
        if i < stage+1:
            ori_bn_names.append(layer_names[i])
        else:
            idm_bn_names.append(layer_names[i])
    return idm_bn_names

def create_model(args):
    # Get use_gem flag from args (defaults to False for backward compatibility)
    use_gem = getattr(args, 'use_gem', False)

    # Base arguments that all models support
    model_kwargs = {
        'num_features': args.features,
        'norm': False,
        'dropout': args.dropout,
        'num_classes': args.nclass,
        'use_gem': use_gem,
    }

    # ViT blocks are only supported by ResNetIBN-based models (e.g., resnet_ibn50a_idm)
    # Check if this is an IBN-based architecture
    is_ibn_arch = 'ibn' in args.arch.lower()

    if is_ibn_arch:
        # Get ViT blocks flags from args (defaults to False for backward compatibility)
        use_vit_blocks = getattr(args, 'use_vit_blocks', False)
        vit_num_blocks = getattr(args, 'vit_num_blocks', 2)
        vit_num_heads = getattr(args, 'vit_num_heads', 4)
        vit_mlp_ratio = getattr(args, 'vit_mlp_ratio', 4.0)

        # Add ViT-related arguments only for IBN architectures
        model_kwargs.update({
            'use_vit_blocks': use_vit_blocks,
            'vit_num_blocks': vit_num_blocks,
            'vit_num_heads': vit_num_heads,
            'vit_mlp_ratio': vit_mlp_ratio,
        })

    # Create model with only supported arguments
    model = models.create(args.arch, **model_kwargs)

    idm_bn_names = filter_layers(args.stage)
    convert_dsbn_idm(model, idm_bn_names, idm=False)

    # ---- Device handling: GPU if available, otherwise CPU ----
    if torch.cuda.is_available():
        print("Using CUDA for model (DataParallel).")
        model = model.cuda()
        model = nn.DataParallel(model)
    else:
        print("CUDA not available; using CPU only (no DataParallel).")

    return model

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters>0) else None
    # Source-free mode: adapt without source images
    source_free_mode = getattr(args, 'source_free', False)
    
    if source_free_mode:
        print("==> Source-free adaptation mode (no source images)")
        # For source-free, we still need a dummy source dataset for compatibility
        # but it won't be used in training (only target is used)
        dataset_source = get_data(args.dataset_source, args.data_dir)
        # Use minimal source classes (just for compatibility)
        source_classes = 0  # Will be set from checkpoint if available
    else:
        print("==> Load source-domain dataset(s)")
        # Check if multi-source training is enabled
        if hasattr(args, 'sources') and args.sources and len(args.sources) > 1:
            print("  Multi-source mode: {}".format(args.sources))
            dataset_source = get_data(args.sources[0], args.data_dir, source_names=args.sources)
        else:
            # Single-source mode (backward compatible)
            dataset_source = get_data(args.dataset_source, args.data_dir)
            source_classes = dataset_source.num_train_pids
    
    print("==> Load target-domain dataset")
    dataset_target = get_data(args.dataset_target, args.data_dir)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)
    
    if not source_free_mode:
        train_loader_source = get_train_loader(args, dataset_source, args.height, args.width,
                                               args.batch_size, args.workers, args.num_instances, iters)
    else:
        # Create empty/minimal source loader for source-free mode
        # The trainer will handle source-free mode internally
        train_loader_source = None

    # Set source_classes (may be 0 for source-free mode)
    if not source_free_mode:
        source_classes = dataset_source.num_train_pids
    else:
        # For source-free, source_classes might come from checkpoint
        # or be set to 0 (target-only training)
        if hasattr(args, 'source_classes') and args.source_classes > 0:
            source_classes = args.source_classes
        else:
            source_classes = 0
    
    args.nclass = source_classes + len(dataset_target.train)
    args.s_class = source_classes
    args.t_class = len(dataset_target.train)

    # Create model
    model = create_model(args)
    print(model)

    # Load checkpoint if provided (for source-free adaptation)
    if args.resume:
        print("==> Loading checkpoint from {}".format(args.resume))
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        if 'best_mAP' in checkpoint:
            best_mAP = checkpoint['best_mAP']
        print("==> Loaded checkpoint (epoch {}, best mAP: {:.1%})".format(start_epoch, best_mAP))

    # Create CBM (Cross-Batch Memory) - internally uses XBM for backward compatibility
    if source_free_mode:
        # For source-free, only use target data for XBM
        datasetSize = len(dataset_target.train)
    else:
        datasetSize = len(dataset_source.train)+len(dataset_target.train)

    args.memorySize = int(args.ratio*datasetSize)
    xbm = XBM(args.memorySize, args.featureSize)  # CBM (Cross-Batch Memory) - using XBM internally
    print('CBM (Cross-Batch Memory) memory size = ', args.memorySize)
    
        # Initialize source-domain class centroids (skip for source-free mode)
    if not source_free_mode and source_classes > 0:
        sour_cluster_loader = get_test_loader(
            dataset_source, args.height, args.width,
            args.batch_size, args.workers,
            testset=sorted(dataset_source.train)
        )
        source_features, _ = extract_features(model, sour_cluster_loader, print_freq=50)
        sour_fea_dict = collections.defaultdict(list)
        for f, pid, _ in sorted(dataset_source.train):
            sour_fea_dict[pid].append(source_features[f].unsqueeze(0))
        source_centers = [torch.cat(sour_fea_dict[pid], 0).mean(0) for pid in sorted(sour_fea_dict.keys())]
        source_centers = torch.stack(source_centers, 0)
        source_centers = F.normalize(source_centers, dim=1)

        # Initialize classifier weights with source centers
        classifier = model.module.classifier if hasattr(model, 'module') else model.classifier
        if torch.cuda.is_available():
            source_centers = source_centers.cuda()
        classifier.weight.data[0:source_classes].copy_(source_centers)

        del source_centers, sour_cluster_loader, sour_fea_dict

    elif source_free_mode:
        print("==> Source-free mode: skipping source centroid initialization")


    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer setup
    # For AIDA runs, we use AdamW + Cosine Annealing + SAM, following the paper's implementation.
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    
    # Get optimizer configuration
    optimizer_type = getattr(args, 'optimizer', 'adam').lower()  # 'adam' or 'adamw'
    use_sam = getattr(args, 'use_sam', False)
    sam_rho = getattr(args, 'sam_rho', 0.05)
    sam_adaptive = getattr(args, 'sam_adaptive', False)
    lr_sched_type = getattr(args, 'lr_sched', 'step').lower()  # 'step' or 'cosine'
    
    # Create base optimizer
    if optimizer_type == 'adamw':
        base_optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        print(f'Using AdamW optimizer with lr={args.lr}, weight_decay={args.weight_decay}')
    else:
        # Default: Adam (backward compatible)
        base_optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        print(f'Using Adam optimizer with lr={args.lr}, weight_decay={args.weight_decay}')
    
    # Wrap with SAM if enabled
    if use_sam:
        from idm.utils.optim_sam import SAM
        optimizer = SAM(params, base_optimizer, rho=sam_rho, adaptive=sam_adaptive)
        print(f'  Wrapped with SAM: rho={sam_rho}, adaptive={sam_adaptive}')
        # For SAM, scheduler attaches to base_optimizer
        scheduler_optimizer = base_optimizer
    else:
        optimizer = base_optimizer
        scheduler_optimizer = optimizer
    
    # Create learning rate scheduler
    if lr_sched_type == 'cosine':
        # Cosine annealing: T_max is total number of iterations (epochs * iters_per_epoch)
        total_iters = args.epochs * args.iters if args.iters > 0 else args.epochs * 400
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            scheduler_optimizer, T_max=total_iters, eta_min=0
        )
        print(f'Using CosineAnnealingLR scheduler: T_max={total_iters}')
    else:
        # Default: StepLR (backward compatible)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            scheduler_optimizer, step_size=args.step_size, gamma=0.1
        )
        print(f'Using StepLR scheduler: step_size={args.step_size}, gamma=0.1')

    # Trainer
    # Determine feature channel size (default to 2048 for ResNet)
    channel_size = args.features if args.features > 0 else 2048
    num_sources = len(args.sources) if hasattr(args, 'sources') and args.sources and len(args.sources) > 1 else 3
    
    trainer = IDM_Trainer(
        model, xbm, args.nclass, 
        margin=args.margin, mu1=args.mu1, mu2=args.mu2, mu3=args.mu3,
        use_ms_idg=getattr(args, 'use_ms_idg', False),
        use_pmr=getattr(args, 'use_pmr', False),
        use_dfc=getattr(args, 'use_dfc', False),
        lambda_idg_0=getattr(args, 'lambda_idg_0', 1.0),
        lambda_pmr_0=getattr(args, 'lambda_pmr_0', 1.0),
        beta_dfc=getattr(args, 'beta_dfc', 0.1),
        lambda_max=getattr(args, 'lambda_max', 10.0),
        lambda_sd=getattr(args, 'lambda_sd', 1.0),
        num_sources=num_sources,
        channel=channel_size
    )

    for epoch in range(args.epochs):
        with torch.no_grad():
            tgt_cluster_loader = get_test_loader(dataset_target, args.height, args.width,
                                                 args.batch_size, args.workers, testset=sorted(dataset_target.train))
            time.sleep(0.5)
            target_features, _ = extract_features(model, tgt_cluster_loader, print_freq=50)
            target_features = torch.cat([target_features[f].unsqueeze(0) for f, _, _ in
                                         sorted(dataset_target.train)], 0)

            del tgt_cluster_loader
            print('==> Create pseudo labels for unlabeled target domain with DBSCAN clustering')

            rerank_dist = compute_jaccard_distance(target_features, k1=args.k1, k2=args.k2, use_gpu=False).numpy()
            print('Clustering and labeling...')
            eps = args.eps
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
            labels = cluster.fit_predict(rerank_dist)
            del rerank_dist
            num_ids = len(set(labels)) - (1 if -1 in labels else 0)
            args.t_class = num_ids

            print('\n Clustered into {} classes \n'.format(args.t_class))


        # generate new dataset and calculate cluster centers
        new_dataset = []
        cluster_centers = collections.defaultdict(list)
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_target.train), labels)):
            if label == -1: continue
            new_dataset.append((fname, source_classes+label, cid))
            cluster_centers[label].append(target_features[i])

        # dataset_target.train = new_dataset

        cluster_centers = [torch.stack(cluster_centers[idx]).mean(0) for idx in sorted(cluster_centers.keys())]
        cluster_centers = torch.stack(cluster_centers)
                # Initialize classifier weights for target-domain pseudo classes
        classifier = model.module.classifier if hasattr(model, 'module') else model.classifier

        cluster_centers = F.normalize(cluster_centers, dim=1).float()
        if torch.cuda.is_available():
            cluster_centers = cluster_centers.cuda()

        classifier.weight.data[args.s_class:args.s_class + args.t_class].copy_(cluster_centers)

    
        del cluster_centers, target_features

        train_loader_target = get_train_loader(args, dataset_target, args.height, args.width,
                                               args.batch_size, args.workers, args.num_instances, iters,
                                               trainset=new_dataset)

        time.sleep(0.5)
        if train_loader_source is not None:
            train_loader_source.new_epoch()
        time.sleep(0.5)
        train_loader_target.new_epoch()
        time.sleep(0.5)
        
        # For source-free mode, create a dummy source loader that returns empty batches
        if source_free_mode and train_loader_source is None:
            # Create a minimal dummy source loader (won't be used in source-free training)
            from idm.utils.data import IterLoader
            from torch.utils.data import DataLoader, TensorDataset
            dummy_data = torch.zeros(1, 3, args.height, args.width)
            dummy_labels = torch.zeros(1, dtype=torch.long)
            dummy_dataset = TensorDataset(dummy_data, dummy_labels)
            dummy_loader = DataLoader(dummy_dataset, batch_size=1, shuffle=False)
            train_loader_source = IterLoader(dummy_loader, length=args.iters)
            # Note: In source-free mode, the trainer should handle this by not using source data
        
        trainer.train(epoch, train_loader_source, train_loader_target, args.s_class, args.t_class, optimizer, 
                      print_freq=args.print_freq, train_iters=args.iters, use_xbm=args.use_xbm, stage=args.stage,
                      source_free=source_free_mode, lr_scheduler=lr_scheduler if lr_sched_type == 'cosine' else None)
                      
        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):

            print('Test on target: ', args.dataset_target)
            _, mAP = evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)
            is_best = (mAP>best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))
        
        # Step learning rate scheduler
        # For CosineAnnealingLR, step per iteration (handled in trainer)
        # For StepLR, step per epoch
        if lr_sched_type == 'step':
            lr_scheduler.step()

        lr_scheduler.step()

    print ('==> Test with the best model on the target domain:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    
    # Check if extended metrics should be computed
    compute_extended = getattr(args, 'compute_extended_metrics', False)
    compute_fid = getattr(args, 'compute_fid', False)
    
    evaluator.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, 
                      cmc_flag=True, 
                      compute_extended=compute_extended,
                      compute_fid=compute_fid)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on UDA re-ID")
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='dukemtmc',
                        help='Single source dataset name (used when --sources not provided)')
    parser.add_argument('--sources', type=str, nargs='+', default=None,
                        help='Multi-source training: list of source dataset names (e.g., market1501 dukemtmc msmt17). '
                             'If provided, overrides --dataset-source.')
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--nclass', type=int, default=1000,
                        help="number of classes (source+target)")
    parser.add_argument('--s-class', type=int, default=1000,
                        help="number of classes (source)")
    parser.add_argument('--t-class', type=int, default=1000,
                        help="number of classes (target)")
    # loss
    parser.add_argument('--margin', type=float, default=0.3,
                        help="margin for triplet loss")
    parser.add_argument('--mu1', type=float, default=0.7,
                        help="weight for loss_bridge_pred")
    parser.add_argument('--mu2', type=float, default=0.1,
                        help="weight for loss_bridge_feat")
    parser.add_argument('--mu3', type=float, default=1,
                        help="weight for loss_div")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50_idm',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--use-gem', '--MODEL.USE_GEM', action='store_true', dest='use_gem',
                        help='Use Generalized Mean Pooling (GeM) instead of Global Average Pooling (GAP)')
    parser.add_argument('--use-vit-blocks', '--MODEL.USE_VIT_BLOCKS', action='store_true', dest='use_vit_blocks',
                        help='Use lightweight ViT blocks on top of CNN feature maps')
    parser.add_argument('--vit-num-blocks', '--MODEL.VIT_NUM_BLOCKS', type=int, default=2, dest='vit_num_blocks',
                        help='Number of ViT blocks to stack (default: 2)')
    parser.add_argument('--vit-num-heads', '--MODEL.VIT_NUM_HEADS', type=int, default=4, dest='vit_num_heads',
                        help='Number of attention heads per ViT block (default: 4)')
    parser.add_argument('--vit-mlp-ratio', '--MODEL.VIT_MLP_RATIO', type=float, default=4.0, dest='vit_mlp_ratio',
                        help='Ratio of MLP hidden dimension to embed_dim (default: 4.0)')
    # SAM optimizer parameters
    parser.add_argument('--use-sam', '--AIDA.USE_SAM', action='store_true', dest='use_sam',
                        help='Use SAM (Sharpness-Aware Minimization) optimizer')
    parser.add_argument('--sam-rho', '--AIDA.SAM_RHO', type=float, default=0.05, dest='sam_rho',
                        help='SAM neighborhood size (rho) (default: 0.05)')
    parser.add_argument('--sam-adaptive', '--AIDA.SAM_ADAPTIVE', action='store_true', dest='sam_adaptive',
                        help='Use adaptive rho per parameter in SAM (default: False)')

    # xbm parameters
    parser.add_argument('--memorySize', type=int, default=8192,
                        help='meomory bank size')
    parser.add_argument('--ratio', type=float, default=1,
                        help='memorySize=ratio*data_size')
    parser.add_argument('--featureSize', type=int, default=2048)
    parser.add_argument('--use-xbm', action='store_true',
                        help="Enable CBM (Cross-Batch Memory) for contrastive learning. If True: strong baseline; if False: naive baseline")

    # idm parameters
    parser.add_argument('--stage', type=int, default=0,
                        help="insert IDM module after stage 0/1/2/3/4")
    # AIDA/PMR parameters
    parser.add_argument('--lambda-sd', '--AIDA.LAMBDA_SD', type=float, default=1.0,
                        dest='lambda_sd',
                        help='Weight for KL divergence term in PMR loss (AIDA.LAMBDA_SD)')
    # AIDA/DFC parameters
    parser.add_argument('--lambda-idg-0', '--AIDA.LAMBDA_IDG_0', type=float, default=1.0,
                        dest='lambda_idg_0',
                        help='Base weight for IDG loss (AIDA.LAMBDA_IDG_0)')
    parser.add_argument('--lambda-pmr-0', '--AIDA.LAMBDA_PMR_0', type=float, default=1.0,
                        dest='lambda_pmr_0',
                        help='Base weight for PMR loss (AIDA.LAMBDA_PMR_0)')
    parser.add_argument('--lambda-dfc-0', '--AIDA.LAMBDA_DFC_0', type=float, default=1.0,
                        dest='lambda_dfc_0',
                        help='Base weight for DFC loss (AIDA.LAMBDA_DFC_0)')
    parser.add_argument('--beta-dfc', '--AIDA.BETA_DFC', type=float, default=0.1,
                        dest='beta_dfc',
                        help='Scaling factor for gradient variance in DFC (AIDA.BETA_DFC)')
    parser.add_argument('--lambda-max', type=float, default=10.0,
                        help='Maximum allowed weight value for DFC clamping')
    parser.add_argument('--use-entropy-modulation', action='store_true',
                        help='Enable entropy-based modulation in DFC')
    parser.add_argument('--entropy-weight', type=float, default=0.5,
                        help='Weight for entropy modulation in DFC (0-1)')
    # DFC loss hyperparameters
    parser.add_argument('--entropy-target', '--AIDA.ENTROPY_TARGET', type=float, default=2.0,
                        dest='entropy_target',
                        help='Target entropy value for DFC loss (AIDA.ENTROPY_TARGET)')
    parser.add_argument('--gradvar-target', '--AIDA.GRADVAR_TARGET', type=float, default=0.1,
                        dest='gradvar_target',
                        help='Target gradient variance for DFC loss (AIDA.GRADVAR_TARGET)')
    parser.add_argument('--w-entropy', '--AIDA.W_ENTROPY', type=float, default=0.1,
                        dest='w_entropy',
                        help='Weight for entropy term in DFC loss (AIDA.W_ENTROPY)')
    parser.add_argument('--w-grad', '--AIDA.W_GRAD', type=float, default=0.1,
                        dest='w_grad',
                        help='Weight for gradient variance term in DFC loss (AIDA.W_GRAD)')
    # Extended evaluation metrics
    parser.add_argument('--compute-extended-metrics', action='store_true',
                        help='Compute extended metrics (NMI, Silhouette, FID) during evaluation')
    parser.add_argument('--compute-fid', action='store_true',
                        help='Compute FID metric (requires --compute-extended-metrics)')
    # AIDA feature flags
    parser.add_argument('--use-ms-idg', '--AIDA.USE_MS_IDG', action='store_true',
                        dest='use_ms_idg',
                        help='Enable Multi-Source Intermediate Domain Generator (AIDA.USE_MS_IDG)')
    parser.add_argument('--use-pmr', '--AIDA.USE_PMR', action='store_true',
                        dest='use_pmr',
                        help='Enable Pseudo-Mirror Regularization (AIDA.USE_PMR)')
    parser.add_argument('--use-dfc', '--AIDA.USE_DFC', action='store_true',
                        dest='use_dfc',
                        help='Enable Dynamic Feedback Controller (AIDA.USE_DFC)')
    # Checkpoint and source-free
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from (for source-free adaptation)')
    parser.add_argument('--source-free', action='store_true',
                        help='Source-free mode: adapt pre-trained model without source images')
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--eval-step', type=int, default=1)

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main()
