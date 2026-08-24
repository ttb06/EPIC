import torch
from collections import defaultdict


class Evaluator:

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {'recall': self.recall_at_k, 'ndcg': self.ndcg_at_k}
        self.evaluation_target = str(
            config.get('evaluation_target', 'sid')).lower()
        if self.evaluation_target not in {'sid', 'item'}:
            raise ValueError(
                'evaluation_target must be either \"sid\" or \"item\", got '
                f'{self.evaluation_target!r}')

        self.sid_to_unique_item = {}
        if self.evaluation_target == 'item':
            sid_groups = defaultdict(list)
            for item, tokens in tokenizer.item2tokens.items():
                sid_groups[tuple(tokens)].append(tokenizer.item2id[item])
            self.sid_to_unique_item = {
                sid: item_ids[0] if len(item_ids) == 1 else None
                for sid, item_ids in sid_groups.items()
            }

        self.eos_token = self.tokenizer.eos_token
        self.maxk = max(config['topk'])

        self.maxk_eval = max(config['val_topk'])

    def calculate_pos_index(self, preds, labels, split):
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()

        if split == 'val':
            assert preds.shape[
                1] == self.maxk_eval, f"preds.shape[1] = {preds.shape[1]} != {self.maxk}"
        elif split == 'test':
            assert preds.shape[
                1] == self.maxk, f"preds.shape[1] = {preds.shape[1]} != {self.maxk}"
        else:
            raise ValueError(f"Invalid split: {split}")

        pos_index = torch.zeros((preds.shape[0], self.maxk), dtype=torch.bool)
        for i in range(preds.shape[0]):
            maxk = self.maxk if (split == 'test') else self.maxk_eval
            if self.evaluation_target == 'item':
                cur_label = int(labels[i].reshape(-1)[0].item())
                for j in range(maxk):
                    predicted_item = self.sid_to_unique_item.get(
                        tuple(preds[i, j].tolist()))
                    if predicted_item == cur_label:
                        pos_index[i, j] = True
                        break
            else:
                cur_label = labels[i].tolist()
                if self.eos_token in cur_label:
                    eos_pos = cur_label.index(self.eos_token)
                    cur_label = cur_label[:eos_pos]
                for j in range(maxk):
                    cur_pred = preds[i, j].tolist()
                    if cur_pred == cur_label:
                        pos_index[i, j] = True
                        break
        return pos_index

    def recall_at_k(self, pos_index, k):
        return pos_index[:, :k].sum(dim=1).cpu().float()

    def ndcg_at_k(self, pos_index, k):
        # Assume only one ground truth item per example
        ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
        dcg = 1.0 / torch.log2(ranks + 1)
        dcg = torch.where(pos_index, dcg, 0)
        return dcg[:, :k].sum(dim=1).cpu().float()

    def calculate_metrics(self, preds, labels, split):
        results = {}
        pos_index = self.calculate_pos_index(preds, labels, split)
        for metric in self.config['metrics']:
            topk_list = self.config['topk'] if (
                split == 'test') else self.config['val_topk']
            # for k in self.config['topk']:
            for k in topk_list:
                results[f"{metric}@{k}"] = self.metric2func[metric](pos_index,
                                                                    k)
        return results
