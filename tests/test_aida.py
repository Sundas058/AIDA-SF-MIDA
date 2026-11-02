"""Numerical, protocol, data, and optimizer regression checks."""
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
from idm.configs.aida_configs import get_config
from idm.data import DomainIdentityBatchSampler, merge_sources
from idm.datasets.reid import ReIDDataset, load_dataset
from idm.engine import validate_config, replace_classifier
from idm.evaluators import retrieval_metrics
from idm.loss.aida import PseudoMirrorLoss, batch_hard_triplet
from idm.models.aida import MSIDG, target_mirror
from idm.trainers import AIDATrainer
from idm.utils.dfc import DynamicFeedbackController
from idm.clustering import cluster_target
from idm.utils.optim_sam import SAM


class EquationTests(unittest.TestCase):
    def test_multisource_statistics(self):
        f = torch.tensor([[[[0., 2.]]], [[[4., 8.]]]], requires_grad=True)
        mixed = MSIDG()(f, torch.tensor([0, 1]), torch.tensor([0.25, 0.75]))
        # Domain means 1 and 6, standard deviations 1 and 2.
        expected = torch.tensor([[[[3., 6.5]]], [[[3., 6.5]]]])
        torch.testing.assert_close(mixed, expected, atol=3e-6, rtol=1e-5)
        mixed.square().sum().backward()
        self.assertTrue(torch.isfinite(f.grad).all())

    def test_missing_source_is_rejected(self):
        with self.assertRaises(ValueError):
            MSIDG()(torch.rand(4, 2, 2, 2), torch.zeros(4, dtype=torch.long), torch.tensor([0.5, 0.5]))

    def test_pmr_normalized_sum_and_pair_geometry(self):
        z = torch.tensor([[1., 0.], [0., 1.]])
        mirror = torch.tensor([[2., 0.], [1., 0.]])
        # Pointwise = (0 + 2)/2; only off-diagonal distance difference = sqrt(2).
        self.assertAlmostEqual(PseudoMirrorLoss()(z, mirror).item(), 1 + math.sqrt(2), places=5)
        self.assertEqual(PseudoMirrorLoss()(z, z * 3).item(), 0)
        self.assertAlmostEqual(PseudoMirrorLoss(identity=False)(z, mirror).item(), math.sqrt(2), places=5)

    def test_triplet_excludes_self_and_invalid_anchors(self):
        z = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]], requires_grad=True)
        self.assertEqual(batch_hard_triplet(z, torch.tensor([0, 0, 1, 1])).item(), 0)
        loss = batch_hard_triplet(z, torch.arange(4))
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_dfc_equations_and_state_roundtrip(self):
        dfc = DynamicFeedbackController(num_sources=2)
        dfc.update(5, 1, torch.tensor([0, 1]), torch.tensor([2., 8.]))
        self.assertAlmostEqual(dfc.lambda_pmr, 1.25)
        dfc.update(5, 3, torch.tensor([0, 1]), torch.tensor([2., 8.]))
        self.assertAlmostEqual(dfc.lambda_idg, 1.375)
        expected = torch.tensor([1.375 * 0.2, 1.375 * 0.8]).softmax(0)
        torch.testing.assert_close(dfc.alpha, expected)
        restored = DynamicFeedbackController(num_sources=2)
        restored.load_state_dict(dfc.state_dict())
        self.assertEqual(restored.state_dict(), dfc.state_dict())
        dfc.update(1e4, 5)
        self.assertEqual(dfc.lambda_pmr, 10)
        self.assertEqual(dfc.scale, 1.5)

    def test_dfc_rolling_window_and_finite_feedback(self):
        dfc = DynamicFeedbackController(window_size=3)
        for i in range(7):
            dfc.update(2, i)
        self.assertEqual(list(dfc.grad_norms), [4., 5., 6.])
        self.assertEqual(len(dfc.variances), 3)
        with self.assertRaises(ValueError):
            dfc.update(float("nan"), 0)

    def test_target_mirror_has_no_source_dependency(self):
        f = torch.randn(8, 3, 2, 2, requires_grad=True)
        result = target_mirror(f)
        self.assertEqual(result.shape, f.shape)
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(f.grad).all())

    def test_clustering_uses_jaccard_and_discards_noise(self):
        # Two well-separated groups with enough intra-cluster neighbors.
        z = torch.cat([torch.tensor([[1., 0.]]).repeat(8, 1),
                       torch.tensor([[0., 1.]]).repeat(8, 1), torch.tensor([[-1., -1.]])])
        labels, centers = cluster_target(z, eps=0.6, min_samples=4, k1=7, k2=1)
        self.assertEqual(len(centers), 2)
        self.assertEqual(labels[-1], -1)
        self.assertEqual(len(set(labels[:8])), 1)
        self.assertNotEqual(labels[0], labels[8])
        expanded_labels, expanded_centers = cluster_target(z, eps=0.6, min_samples=4, k1=7, k2=3)
        self.assertEqual(len(expanded_centers), 2)
        self.assertEqual(len(set(expanded_labels[:8])), 1)


class ProtocolTests(unittest.TestCase):
    def test_target_exclusion_and_sf_requires_checkpoint(self):
        config = get_config("market_duke_to_msmt")
        config["target"] = "market1501"
        with self.assertRaises(ValueError):
            validate_config(config, "dg")
        sf = get_config("adapt_to_market", "sf_adapt")
        with self.assertRaises(ValueError):
            validate_config(sf, "sf_adapt")

    def test_configs_are_independent(self):
        a = get_config("market_duke_to_msmt")
        a["sources"].append("personx")
        self.assertEqual(len(get_config("market_duke_to_msmt")["sources"]), 2)

    def test_source_identity_offsets(self):
        first = ReIDDataset([("a", 0, 0), ("b", 1, 0)], [], [])
        second = ReIDDataset([("c", 0, 1), ("d", 1, 1)], [], [])
        rows, classes = merge_sources([first, second])
        self.assertEqual(classes, 4)
        self.assertEqual([r[1] for r in rows], [0, 1, 2, 3])
        sampler = DomainIdentityBatchSampler(rows, batch_size=8, instances=2, steps=3)
        for batch in sampler:
            self.assertEqual(len(batch), 8)
            self.assertEqual({rows[i][3] for i in batch}, {0, 1})

    def test_train_reader_does_not_require_test_splits(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "market1501" / "bounding_box_train"
            root.mkdir(parents=True)
            Image.new("RGB", (16, 32)).save(root / "0007_c1_001.jpg")
            dataset = load_dataset("market1501", directory, splits=("train",))
            self.assertEqual(dataset.train[0][1], 0)
            self.assertEqual(dataset.query, [])
            unlabeled = load_dataset("market1501", directory, splits=("train",), unlabeled_train=True)
            self.assertEqual(unlabeled.train[0][1], -1)

    def test_camera_filtering_and_average_precision(self):
        q = torch.tensor([[1., 0.]])
        g = torch.tensor([[1., 0.], [0.9, 0.1], [0., 1.]])
        metrics = retrieval_metrics(q, g, [("q", 1, 0)],
                                    [("self", 1, 0), ("other", 2, 1), ("positive", 1, 1)])
        self.assertEqual(metrics["mAP"], 50)
        self.assertEqual(metrics["rank1"], 0)
        self.assertEqual(metrics["rank5"], 100)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU())
        self.classifier = nn.Linear(4, 4, bias=False)

    def feature_maps(self, x):
        return self.backbone(x)

    def embed(self, x):
        return torch.nn.functional.normalize(x.mean((2, 3)), dim=1)

    def reset_classifier(self, centers):
        self.classifier = nn.Linear(4, len(centers), bias=False)
        with torch.no_grad():
            self.classifier.weight.copy_(centers)


class OptimizerTests(unittest.TestCase):
    def test_classifier_refresh_preserves_backbone_optimizer_state(self):
        model = TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        z = model.embed(model.feature_maps(torch.rand(4, 3, 4, 4)))
        model.classifier(z).sum().backward()
        optimizer.step()
        backbone_parameter = model.backbone[0].weight
        old_classifier = model.classifier.weight
        old_state = optimizer.state[backbone_parameter]["exp_avg"].clone()
        replace_classifier(model, torch.rand(5, 4), optimizer)
        self.assertEqual(model.classifier.out_features, 5)
        torch.testing.assert_close(optimizer.state[backbone_parameter]["exp_avg"], old_state)
        self.assertNotIn(old_classifier, optimizer.state)
        self.assertTrue(any(p is model.classifier.weight for p in optimizer.param_groups[0]["params"]))

    def test_joint_uda_bridge_and_mirror_gradients(self):
        from idm.models.idm_module import IDM
        model = TinyModel()
        model.idm = IDM(channel=4)
        config = get_config("market1501_to_dukemtmc", "uda")
        controller = DynamicFeedbackController(num_sources=2)
        trainer = AIDATrainer(model, config, controller)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        source = (torch.rand(4, 3, 4, 4), [], torch.tensor([0, 0, 1, 1]),
                  torch.zeros(4), torch.zeros(4, dtype=torch.long))
        target = (torch.rand(4, 3, 4, 4), [], torch.tensor([2, 2, 3, 3]),
                  torch.zeros(4), torch.zeros(4, dtype=torch.long))
        values = trainer.step(source, optimizer, target_batch=target)
        self.assertTrue(np.isfinite(values["loss"]))
        self.assertIsNotNone(model.idm.adaptiveFC3.weight.grad)
        self.assertTrue(torch.isfinite(model.idm.adaptiveFC3.weight.grad).all())
        self.assertEqual(controller.updates, 1)

    def test_sam_dfc_updates_once_and_bn_only_once(self):
        torch.manual_seed(7)
        model = TinyModel()
        config = get_config("market_duke_to_msmt")
        controller = DynamicFeedbackController(num_sources=2)
        trainer = AIDATrainer(model, config, controller)
        base = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer = SAM(model.parameters(), base)
        batch = (torch.rand(8, 3, 4, 4), [], torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
                 torch.zeros(8), torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]))
        before = model.backbone[0].weight.detach().clone()
        values = trainer.step(batch, optimizer)
        self.assertTrue(np.isfinite(values["loss"]))
        self.assertEqual(controller.updates, 1)
        self.assertEqual(model.backbone[1].num_batches_tracked.item(), 1)
        self.assertFalse(torch.equal(before, model.backbone[0].weight))

    def test_source_free_step_uses_pseudo_labels(self):
        model = TinyModel()
        config = get_config("adapt_to_market", "sf_adapt")
        controller = DynamicFeedbackController()
        trainer = AIDATrainer(model, config, controller)
        optimizer = torch.optim.AdamW(model.parameters())
        batch = (torch.rand(8, 3, 4, 4), [], torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
                 torch.zeros(8), torch.zeros(8, dtype=torch.long))
        values = trainer.step(batch, optimizer, stage="sf")
        self.assertEqual(values["idg"], 0)
        self.assertEqual(controller.updates, 1)


if __name__ == "__main__":
    unittest.main()

