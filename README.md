# EPIC

Official anonymous implementation of **EPIC: Explicit Posterior Item
Conditioning for Semantic-ID Diffusion Recommendation**.

EPIC augments a frozen masked-diffusion recommender with explicit item-level
reasoning during denoising. At every step it builds the set of catalog items
compatible with the current partial Semantic ID, learns a normalized posterior
over a retained candidate support, and marginalizes that posterior back to the
unresolved SID positions. The item signal therefore changes the generation
trajectory while the target is still reachable; it is not merely a post-hoc
reranker.

## Repository contents

- `LLaDA/` and `genrec/`: the LLaDA-Rec backbone plus the EPIC adapter.
- `configs/lladarec/`: reproducible backbone-training recipes.
- `configs/epic/`: the final EPIC recipes used for the four benchmarks.
- `semantic_ids/`: fixed H4 Semantic IDs used by the released recipes.
- `tests/`: checks for configuration parsing and collision-safe item metrics.

Exploratory sweeps, retired variants, checkpoints, logs, private paths, and
draft paper material are intentionally excluded.

## Environment

Python 3.10 and a CUDA-capable PyTorch installation are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`flash-attn` is optional. The default configuration runs without it.

## Data

The experiments use the 5-core Amazon Reviews benchmarks. Download the
Amazon-2014 Beauty, Sports and Outdoors, and Toys and Games review/metadata
files from the [UCSD Amazon review page](https://jmcauley.ucsd.edu/data/amazon/index_2014.html)
and place them as follows:

```text
data/amazon2014/
├── reviews_Beauty_5.json.gz
├── meta_Beauty.json.gz
├── reviews_Sports_and_Outdoors_5.json.gz
├── meta_Sports_and_Outdoors.json.gz
├── reviews_Toys_and_Games_5.json.gz
└── meta_Toys_and_Games.json.gz
```

For Amazon Reviews 2023 Musical Instruments, place the official train, valid,
test and metadata files under:

```text
cache/AmazonReviews2023/Musical_Instruments/raw/
├── Musical_Instruments.train.csv
├── Musical_Instruments.valid.csv
├── Musical_Instruments.test.csv
└── meta_Musical_Instruments.jsonl
```

The loader can also retrieve the 2023 split when the local `raw/` directory is
absent. Local files are preferable for reproducibility.

## Reproduction workflow

The required order is **Semantic IDs → LLaDA-Rec backbone → EPIC adapter**.
The commands below use Beauty; substitute `sports`, `toys`, or `musical` and
the matching dataset class where appropriate.

### 1. Use or train Semantic IDs

The exact H4 mappings used by this artifact are already included:

```text
semantic_ids/beauty.sem_ids
semantic_ids/sports.sem_ids
semantic_ids/toys.sem_ids
semantic_ids/musical.sem_ids
```

Verify them with:

```bash
sha256sum -c MANIFEST.sha256
```

To train a new Sentence-T5 + MultiHeadVQVAE tokenizer instead:

```bash
python main_token.py \
  --dataset AmazonReviews2014 \
  --config configs/lladarec/beauty.yaml \
  --config configs/tokenizer.yaml
```

This produces a `.sem_ids` file under `outputs/tokenizer/`. Training a new
tokenizer defines a new item representation and will not reproduce the table
exactly. The paper-exact Musical mapping was built from a cached 3072-dimensional
embedding file; the checked-in `semantic_ids/musical.sem_ids` avoids requiring
an external embedding API.

### 2. Train LLaDA-Rec first

```bash
python main.py \
  --dataset AmazonReviews2014 \
  --config configs/lladarec/beauty.yaml \
  --sem_ids_path=semantic_ids/beauty.sem_ids
```

The selected backbone is written below `outputs/lladarec/`. Selection uses
validation NDCG@5; the test split must not be used for tuning. Multi-GPU runs
can replace `python` with `accelerate launch`.

For Musical Instruments use:

```bash
python main.py \
  --dataset AmazonReviews2023 \
  --config configs/lladarec/musical.yaml \
  --sem_ids_path=semantic_ids/musical.sem_ids
```

### 3. Train the EPIC adapter

Pass the validation-selected LLaDA-Rec checkpoint from step 2:

```bash
python main.py \
  --dataset AmazonReviews2014 \
  --config configs/epic/beauty.yaml \
  --sem_ids_path=semantic_ids/beauty.sem_ids \
  --init_checkpoint=/absolute/path/to/lladarec_checkpoint.pth
```

The backbone is frozen and only parameters whose names start with
`item_arch_` are optimized. EPIC starts as an exact no-op because its residual
scales and feedback gate are zero-initialized.

### 4. Evaluate an existing EPIC checkpoint

```bash
python main.py \
  --dataset AmazonReviews2014 \
  --config configs/epic/beauty.yaml \
  --sem_ids_path=semantic_ids/beauty.sem_ids \
  --init_checkpoint=/absolute/path/to/epic_checkpoint.pth \
  --eval_only=True
```

For multiple seeds, repeat adapter training with `--rand_seed=<seed>` and a
unique `--run_id=<name>`. Report the mean and sample standard deviation. Model
and hyperparameter selection remains validation-only.

## Configuration guide

The most important EPIC options are:

| Option | Meaning |
|---|---|
| `item_arch_candidate_topk` | Maximum retained candidate support `M`; feasible candidates are proposed by frozen backbone token likelihood. |
| `item_arch_transition_recent_k` | Number of most recent history items used by personalized transition evidence. |
| `item_arch_temperature` | Temperature of the normalized item posterior. |
| `item_arch_feedback_strength` | Scale of the item-to-token residual used inside denoising. |
| `item_arch_loss_min_bucket`, `item_arch_loss_max_bucket` | Identification frontier on which direct item supervision is active. |
| `item_arch_sid_kernel_w` | Weight of the noise-adaptive SID-neighborhood target. |
| `subset_resolved_tokens` | Number of SID positions resolved before exact compatible-item completion. |

`sem_ids_path` must refer to the same item mapping used to train the backbone.
Changing Semantic IDs while reusing a checkpoint is invalid, even when the
number of codebooks and codebook sizes are unchanged.

## Metrics and evaluation protocol

Evaluation is full-catalog, next-item prediction with one held-out target per
user. The last interaction is test, the penultimate interaction is validation,
and earlier interactions form the training sequence.

- **Recall@K** is 1 when the target appears within the first `K` predictions
  and 0 otherwise, averaged over users.
- **NDCG@K** is `1 / log2(rank + 1)` when the target appears at a 1-indexed
  rank at most `K`, and 0 otherwise, averaged over users. With one relevant
  item, NDCG differs from Recall only by the rank discount.
- **Item-level collision rule:** a generated SID receives credit only if it is
  catalog-valid, has a singleton catalog preimage, and maps to the held-out
  item. A SID shared by multiple items receives no credit. This prevents
  optimistic token-level evaluation.
- **Checkpoint selection:** `val_metric: ndcg@5`. Test metrics are computed only
  after selection.

The reported metrics are Recall@5, NDCG@5, Recall@10, and NDCG@10. The code
also logs `@1` as a diagnostic.

## Tests

```bash
python -m compileall -q LLaDA genrec main.py main_token.py
python -m unittest discover -s tests -v
```

## Acknowledgment and citation

This implementation builds on
[LLaDA-Rec](https://github.com/TengShi-RUC/LLaDA-Rec). See `NOTICE.md` for code
provenance. Author identities and the final bibliographic entry are omitted
during double-blind review; `CITATION.cff` should be updated after acceptance.
