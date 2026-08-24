import unittest
from types import SimpleNamespace

import torch

from genrec.models.LLaDARec.model import LLaDARec
from genrec.utils import get_config


class EpicModelInitTest(unittest.TestCase):
    def test_publication_adapter_shape(self):
        config = get_config(
            'LLaDARec',
            'AmazonReviews2014',
            'configs/epic/beauty.yaml',
            {},
        )
        config.update({
            'device': torch.device('cpu'),
            'accelerator': SimpleNamespace(is_main_process=True),
        })
        item2id = {'a': 1, 'b': 2, 'c': 3}
        dataset = SimpleNamespace(n_items=4, item2id=item2id)
        tokenizer = SimpleNamespace(
            n_digit=4,
            item2id=item2id,
            item2tokens={
                'a': (1, 5, 9, 13),
                'b': (2, 6, 10, 14),
                'c': (3, 7, 11, 15),
            },
            mask_token=18,
            eos_token=17,
            vocab_size=19,
            padding_token=0,
            ignored_label=-100,
        )

        model = LLaDARec(config, dataset, tokenizer)
        adapter_parameters = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith('item_arch_')
        )

        self.assertEqual(model.item_arch_mode, 'epic')
        self.assertEqual(adapter_parameters, 391_278)


if __name__ == '__main__':
    unittest.main()
