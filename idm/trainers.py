"""Optimization of source, target-only, and joint source-target objectives."""
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from .models.aida import MSIDG, target_mirror, channel_statistics, adain
from .models.domain_bn import set_domain
from .loss.aida import PseudoMirrorLoss, supervised_loss, batch_hard_triplet


@contextmanager
def preserve_bn_statistics(model):
    snapshots = [(m, m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
                 for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats]
    try:
        yield
    finally:
        with torch.no_grad():
            for m, mean, var, count in snapshots:
                m.running_mean.copy_(mean)
                m.running_var.copy_(var)
                m.num_batches_tracked.copy_(count)


def gradient_norm(model):
    grads = [p.grad.detach().square().sum() for p in model.backbone.parameters() if p.grad is not None]
    return torch.stack(grads).sum().sqrt().item() if grads else 0.0


class AIDATrainer:
    def __init__(self, model, config, controller):
        self.model, self.config, self.controller = model, config, controller
        self.idg = MSIDG()
        self.pmr = PseudoMirrorLoss(config["lambda_rel"], config["pmr_identity"], config["pmr_relational"])

    def losses(self, batch, stage="aida"):
        c, model = self.config, self.model
        device = next(model.parameters()).device
        images, labels, domains = batch[0].to(device), batch[2].to(device), batch[4].to(device)
        maps = model.feature_maps(images)
        z = model.embed(maps)
        logits = model.classifier(z)
        if stage == "sf":
            base = supervised_loss(logits, z, labels, c["margin"], c["lambda_tri"])
        else:
            base = sum(supervised_loss(logits[domains == d], z[domains == d], labels[domains == d],
                                       c["margin"], c["lambda_tri"]) for d in domains.unique())
        idg_loss = pmr_loss = base * 0
        feedback_logits = logits
        if stage != "supervised":
            if stage == "sf":
                mirror = target_mirror(maps)
            elif c["use_ms_idg"]:
                alpha = self.controller.alpha.to(device) if c["use_dfc"] else torch.full(
                    (self.controller.num_sources,), 1 / self.controller.num_sources, device=device)
                mirror = self.idg(maps, domains, alpha)
            else:
                mirror = None
            if mirror is not None:
                mirror_z = model.embed(mirror)
                feedback_logits = model.classifier(mirror_z)
                if stage != "sf" and c["use_ms_idg"]:
                    idg_loss = supervised_loss(feedback_logits, mirror_z, labels, c["margin"], c["lambda_tri"])
                if c["use_pmr"]:
                    pmr_loss = self.pmr(z, mirror_z)
        total = base + self.controller.lambda_idg * idg_loss + self.controller.lambda_pmr * pmr_loss
        entropies = self.controller.prediction_entropy(feedback_logits)
        return total, {"supervised": base, "idg": idg_loss, "pmr": pmr_loss}, entropies, domains

    def uda_losses(self, source, target):
        c, model = self.config, self.model
        device = next(model.parameters()).device
        set_domain(model, "source")
        source_maps = model.feature_maps(source[0].to(device))
        source_z = model.embed(source_maps)
        source_logits = model.classifier(source_z)
        set_domain(model, "target")
        target_maps = model.feature_maps(target[0].to(device))
        target_z = model.embed(target_maps)
        target_logits = model.classifier(target_z)
        ys, yt = source[2].to(device), target[2].to(device)
        base = supervised_loss(source_logits, source_z, ys, c["margin"], c["lambda_tri"])
        base = base + supervised_loss(target_logits, target_z, yt, c["margin"], c["lambda_tri"])
        maps = torch.cat([source_maps, target_maps])
        domains = torch.cat([torch.zeros(len(ys), device=device, dtype=torch.long),
                             torch.ones(len(yt), device=device, dtype=torch.long)])
        mixed_maps, lam = model.idm(maps)
        mixed_maps = mixed_maps[len(maps):]
        # Learned pairwise interpolation, regulated by the detached DFC source weights.
        if c["use_dfc"]:
            lam = lam * self.controller.alpha.to(device)[None, :]
            lam = lam / lam.sum(1, keepdim=True)
            mixed_maps = lam[:, 0, None, None, None] * source_maps + lam[:, 1, None, None, None] * target_maps
        set_domain(model, "target")
        mixed_z = model.embed(mixed_maps)
        mixed_logits = model.classifier(mixed_z)
        bridge_prob = (lam[:, 0] * F.cross_entropy(mixed_logits, ys, reduction="none") +
                       lam[:, 1] * F.cross_entropy(mixed_logits, yt, reduction="none")).mean()
        distance_s = torch.linalg.vector_norm(mixed_z - source_z, dim=1)
        distance_t = torch.linalg.vector_norm(mixed_z - target_z, dim=1)
        bridge_feat = (lam[:, 0] * distance_s + lam[:, 1] * distance_t).mean()
        diversity = -lam.std(0, unbiased=False).sum()
        source_mean, source_std = channel_statistics(source_maps)
        target_mean, target_std = channel_statistics(target_maps)
        mirror = torch.cat([adain(source_maps, target_mean, target_std),
                            adain(target_maps, source_mean, source_std)])
        set_domain(model, "source")
        mirror_s = model.embed(mirror[:len(ys)])
        set_domain(model, "target")
        mirror_t = model.embed(mirror[len(ys):])
        mirror_z = torch.cat([mirror_s, mirror_t])
        mirror_logits = model.classifier(mirror_z)
        idg_loss = c["mu1"] * bridge_prob + c["mu2"] * bridge_feat + c["mu3"] * diversity
        pmr_loss = self.pmr(torch.cat([source_z, target_z]), mirror_z) if c["use_pmr"] else base * 0
        total = base + self.controller.lambda_idg * idg_loss + self.controller.lambda_pmr * pmr_loss
        feedback_logits = torch.cat([source_logits, target_logits])
        return total, {"supervised": base, "idg": idg_loss, "pmr": pmr_loss}, self.controller.prediction_entropy(feedback_logits), domains

    def step(self, batch, optimizer, stage="aida", target_batch=None):
        self.model.train()
        optimizer.zero_grad()
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        compute = lambda: self.uda_losses(batch, target_batch) if target_batch is not None else self.losses(batch, stage)
        loss, components, entropies, domains = compute()
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss is not finite.")
        loss.backward()
        norm = gradient_norm(self.model)
        if hasattr(optimizer, "first_step"):
            optimizer.first_step(zero_grad=True)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            with preserve_bn_statistics(self.model):
                second_loss, _, _, _ = compute()
                if not torch.isfinite(second_loss):
                    raise FloatingPointError("SAM loss is not finite.")
                second_loss.backward()
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.step()
        # Causal feedback: completed gradients regulate the following mini-batch.
        if self.config["use_dfc"] and stage != "supervised":
            self.controller.update(entropies.mean().item(), norm,
                domains if stage != "sf" else None, entropies if stage != "sf" else None)
        return {"loss": float(loss.detach()), **{k: float(v.detach()) for k, v in components.items()},
                "entropy": float(entropies.mean()), "grad_norm": norm,
                "lambda_idg": self.controller.lambda_idg, "lambda_pmr": self.controller.lambda_pmr,
                "alpha": self.controller.alpha.tolist()}

