import torch


class XBM:
    """
    Cross-Batch Memory for single-label features.
    Works on CPU or GPU depending on availability.
    """

    def __init__(self, memory_size, feature_size, device=None):
        self.K = memory_size
        self.D = feature_size

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.feats = torch.zeros(self.K, self.D, device=self.device)
        self.targets = torch.zeros(self.K, dtype=torch.long, device=self.device)
        self.ptr = 0

    @property
    def is_full(self):
        # Memory considered "full" if last slot has been written (non-zero label)
        return self.targets[-1].item() != 0

    def get(self):
        if self.is_full:
            return self.feats, self.targets
        else:
            return self.feats[: self.ptr], self.targets[: self.ptr]

    def enqueue_dequeue(self, feats, targets):
        """
        feats:  [batch_size, D]
        targets: [batch_size]
        """
        # Ensure we are on the same device
        feats = feats.to(self.device)
        targets = targets.to(self.device)

        q_size = len(targets)
        if self.ptr + q_size > self.K:
            # Wrap: overwrite the last q_size slots
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size

    def clean_target_domain(self, source_classes, target_classes):
        """
        Remove samples whose labels fall in the [source_classes, source_classes + target_classes)
        interval (i.e., target-domain classes).
        """
        empty_feats = torch.zeros(self.K, self.D, device=self.device)
        empty_targets = torch.zeros(self.K, dtype=torch.long, device=self.device)

        j = 0
        for i in range(self.K):
            y = self.targets[i].item()
            if source_classes <= y < source_classes + target_classes:
                # Skip target-domain samples
                continue
            else:
                empty_feats[j] = self.feats[i]
                empty_targets[j] = self.targets[i]
                j += 1

        self.feats = empty_feats
        self.targets = empty_targets
        if j > 0:
            self.ptr = j


class MultiLabelXBM:
    """
    Cross-Batch Memory for two labels per feature (source + target labels).
    """

    def __init__(self, memory_size, feature_size, device=None):
        self.K = memory_size
        self.D = feature_size

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.feats = torch.zeros(self.K, self.D, device=self.device)
        self.targets_s = torch.zeros(self.K, dtype=torch.long, device=self.device)
        self.targets_t = torch.zeros(self.K, dtype=torch.long, device=self.device)
        self.ptr = 0

    @property
    def is_full(self):
        return self.targets_s[-1].item() != 0

    def get(self):
        if self.is_full:
            return self.feats, self.targets_s, self.targets_t
        else:
            return (
                self.feats[: self.ptr],
                self.targets_s[: self.ptr],
                self.targets_t[: self.ptr],
            )

    def enqueue_dequeue(self, feats, source_targets, target_targets):
        feats = feats.to(self.device)
        source_targets = source_targets.to(self.device)
        target_targets = target_targets.to(self.device)

        q_size = len(source_targets)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.targets_s[-q_size:] = source_targets
            self.targets_t[-q_size:] = target_targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets_s[self.ptr: self.ptr + q_size] = source_targets
            self.targets_t[self.ptr: self.ptr + q_size] = target_targets
            self.ptr += q_size

    def clean_target_domain(self, source_classes, target_classes):
        """
        Example clean based on source labels (you probably don't use this in training).
        """
        empty_feats = torch.zeros(self.K, self.D, device=self.device)
        empty_targets_s = torch.zeros(self.K, dtype=torch.long, device=self.device)
        empty_targets_t = torch.zeros(self.K, dtype=torch.long, device=self.device)

        j = 0
        for i in range(self.K):
            y = self.targets_s[i].item()
            if source_classes <= y < source_classes + target_classes:
                continue
            else:
                empty_feats[j] = self.feats[i]
                empty_targets_s[j] = self.targets_s[i]
                empty_targets_t[j] = self.targets_t[i]
                j += 1

        self.feats = empty_feats
        self.targets_s = empty_targets_s
        self.targets_t = empty_targets_t
        if j > 0:
            self.ptr = j


class LogitsXBM:
    """
    Cross-Batch Memory for logits instead of features.
    """

    def __init__(self, memory_size, logits_size, device=None):
        self.K = memory_size
        self.D = logits_size

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.feats = torch.zeros(self.K, self.D, device=self.device)
        self.targets = torch.zeros(self.K, dtype=torch.long, device=self.device)
        self.ptr = 0

    @property
    def is_full(self):
        return self.targets[-1].item() != 0

    def get(self):
        if self.is_full:
            return self.feats, self.targets
        else:
            return self.feats[: self.ptr], self.targets[: self.ptr]

    def enqueue_dequeue(self, feats, targets):
        feats = feats.to(self.device)
        targets = targets.to(self.device)

        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size

    def clean_target_domain(self, source_classes, target_classes):
        empty_feats = torch.zeros(self.K, self.D, device=self.device)
        empty_targets = torch.zeros(self.K, dtype=torch.long, device=self.device)

        j = 0
        for i in range(self.K):
            y = self.targets[i].item()
            if source_classes <= y < source_classes + target_classes:
                continue
            else:
                empty_feats[j] = self.feats[i]
                empty_targets[j] = self.targets[i]
                j += 1

        self.feats = empty_feats
        self.targets = empty_targets
        if j > 0:
            self.ptr = j


class CBM:
    """
    Cross-Batch Memory (CBM) wrapper for XBM.
    Thin wrapper to match paper terminology.
    """

    def __init__(self, memory_size, feature_size, device=None):
        self._xbm = XBM(memory_size, feature_size, device=device)

    @property
    def is_full(self):
        return self._xbm.is_full

    def get(self):
        return self._xbm.get()

    def enqueue_dequeue(self, feats, targets):
        self._xbm.enqueue_dequeue(feats, targets)

    def clean_target_domain(self, source_classes, target_classes):
        self._xbm.clean_target_domain(source_classes, target_classes)


