import math
import unittest

import torch

from genrec.evaluator import Evaluator


class FakeTokenizer:
    eos_token = 99
    item2tokens = {
        'collision_a': (1, 2),
        'collision_b': (1, 2),
        'unique_c': (3, 4),
    }
    item2id = {
        'collision_a': 1,
        'collision_b': 2,
        'unique_c': 3,
    }


class EvaluatorTest(unittest.TestCase):
    def setUp(self):
        self.evaluator = Evaluator(
            {
                'evaluation_target': 'item',
                'topk': [1, 2],
                'val_topk': [2],
                'metrics': ['recall', 'ndcg'],
            },
            FakeTokenizer(),
        )

    def test_unique_item_gets_rank_discounted_credit(self):
        predictions = torch.tensor([[[8, 8], [3, 4]]])
        labels = torch.tensor([[3]])
        metrics = self.evaluator.calculate_metrics(
            predictions, labels, split='test')

        self.assertEqual(metrics['recall@1'].item(), 0.0)
        self.assertEqual(metrics['recall@2'].item(), 1.0)
        self.assertAlmostEqual(
            metrics['ndcg@2'].item(), 1.0 / math.log2(3), places=6)

    def test_collided_sid_receives_no_item_level_credit(self):
        predictions = torch.tensor([[[1, 2], [8, 8]]])
        labels = torch.tensor([[1]])
        metrics = self.evaluator.calculate_metrics(
            predictions, labels, split='test')

        self.assertEqual(metrics['recall@2'].item(), 0.0)
        self.assertEqual(metrics['ndcg@2'].item(), 0.0)


if __name__ == '__main__':
    unittest.main()
