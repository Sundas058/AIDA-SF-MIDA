# 🧠 Method-to-code reference

| Paper reference | Computation | Code |
|:--|:--|:--|
| Eqs. (2)–(5), §3.3 | ImageNet ResNet-50, global/part tokens, embedding normalization | [AIDAEmbedding](../idm/models/aida.py) |
| Eqs. (6)–(8) | Cross-entropy and batch-hard Euclidean triplet loss | [supervised_loss](../idm/loss/aida.py) |
| Eqs. (12)–(14) | Channel-statistics transfer, convex domain mixture | [MSIDG and adain](../idm/models/aida.py) |
| Eq. (15) | Mean squared L2 distance between normalized embedding pairs | [PseudoMirrorLoss](../idm/loss/aida.py) |
| Eq. (16) | Mean absolute difference of unique off-diagonal pair distances | [PseudoMirrorLoss](../idm/loss/aida.py) |
| Eq. (17) | Pointwise PMR plus weighted relational PMR | [PseudoMirrorLoss](../idm/loss/aida.py) |
| Eq. (18) | Batch prediction entropy | [prediction_entropy](../idm/utils/dfc.py) |
| Eqs. (19)–(20) | Gradient-norm population variance and rolling-maximum normalization | [DynamicFeedbackController](../idm/utils/dfc.py) |
| Eqs. (21)–(22) | Multiplicative, clipped IDG/PMR weights | [DynamicFeedbackController](../idm/utils/dfc.py) |
| Eqs. (23)–(24) | Bounded feedback scale and softmax source coefficients | [DynamicFeedbackController](../idm/utils/dfc.py) |
| Algorithm 1 | Supervised source stage, AIDA stage, target-only refinement | [engine](../idm/engine.py), [trainer](../idm/trainers.py) |

## Source transitions

For each source sample, channel means and population standard deviations are computed over spatial positions. Each domain's statistics are the mean of its per-image statistics. MS-IDG combines these domain summaries with a single simplex vector and transfers the resulting statistics to each anchor's normalized content.

Every source mini-batch contains all configured domains. The initial mixing vector is uniform. DFC replaces it with a softmax over the most recent per-domain mean entropies.

The IDG loss is identity classification plus batch-hard triplet loss on the intermediate embeddings. PMR compares original and intermediate normalized embeddings. No feature-reconstruction MSE is substituted for intermediate identity supervision.

## Pseudo-mirror losses

The pointwise term sums squared embedding-coordinate differences per sample, then averages over samples. The relational term averages absolute Euclidean-distance differences over unordered, non-self sample pairs. Either component can be switched independently.

Both embedding branches receive gradients. A positive triplet must be a distinct sample with the same identity; anchors without both a positive and a negative contribute zero.

## Feedback timing and state

The controller is detached from autograd. One completed optimization step supplies its prediction entropy and the L2 norm of gradients of backbone parameters. Population variance is computed over the latest 100 such norms. Its normalization denominator is the largest of the latest 100 variance estimates; zero variance maps to zero normalized variance.

Updated loss weights and coefficients apply to the next mini-batch. This makes the loop causal while avoiding a self-dependence between the current loss weights and the current gradient norm.

SAM computes two gradients with fixed controller state and the same random mirror selection. Its second pass restores batch-normalization running buffers afterward. DFC receives the first-pass gradient norm once.

## Global and local interactions

GeM produces the global token. Adaptive horizontal pooling produces four part tokens by default. Separate trainable experience memories, each with 32 slots and learned reliability values, interact with these tokens through normalized similarity attention and a residual gate. The local tokens are averaged and fused with the global token before batch normalization and L2 normalization.

These memory tensors are model parameters. Source images, image paths, and per-image source embeddings are not embedded in checkpoints.

## Target refinement

Target embeddings are re-extracted before every epoch. DBSCAN operates on k-reciprocal Jaccard distances and excludes noise label `-1`. Cluster centers initialize the current target classifier. Refreshing classifier parameters removes their obsolete optimizer state while preserving backbone optimizer state.

Target pseudo-mirrors use a nonzero random cyclic permutation of per-image target statistics. SF-MIDA optimizes pseudo-label classification, triplet, and PMR. It does not add a labeled-source IDG objective.

Checkpoint provenance records source domain names. Target overlap with those names is rejected before dataset loading.

## Additional configuration defaults

These fixed implementation settings are exposed in the configuration:

| Parameter | Default |
|:--|--:|
| Source supervised epochs | 10 |
| Total DG/source-pretraining epochs | 50 |
| Total UDA epochs | 60 |
| Triplet margin | 0.3 |
| Triplet weight | 1.0 |
| Relational PMR weight | 1.0 |
| Identity instances per batch group | 4 |
| Iterations per epoch | 400 |
| SAM radius | 0.05 |
| Horizontal parts | 4 |
| Memory slots per branch | 32 |
| Random-erasing probability | 0.5 |
| Scheduler | Epoch-wise cosine annealing |
| Default seed | 1 |

UDA uses separate source/target normalization and global average pooling, with local/memory branches disabled. Its pairwise IDM learns source/target coefficients. Bridge probability loss uses the corresponding mixed identity labels; feature bridges and interpolation diversity retain their own weights (`mu1=0.7`, `mu2=0.1`, `mu3=1.0`). DFC modulates interpolation weights and the combined bridge objective. Cross-domain AdaIN creates the pseudo-mirror embeddings.

## Retrieval and representation metrics

CMC and mAP use normalized embeddings, remove junk gallery identities, and exclude same-identity/same-camera entries. Queries without a valid positive are omitted from the mean and counted in the output.

Extended NMI uses seeded K-means assignments with the number of evaluation identities. Silhouette uses normalized embeddings, cosine distance, and at most 5,000 evaluation samples. These metrics are computed only when requested during evaluation.

