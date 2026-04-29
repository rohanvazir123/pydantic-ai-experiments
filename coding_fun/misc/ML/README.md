# ML Notebooks

| Notebook | Topic |
|---|---|
| `two_tower_NN.ipynb` | Two-Tower recommendation model trained with triplet (contrastive) loss |
| `minerU_2.5.ipynb` | MinerU 2.5 document parsing |

---

## two_tower_NN.ipynb

A recommender system trained on MovieLens 100k using a Two-Tower neural network and contrastive learning.

### How contrastive learning is done here

Contrastive learning is implemented via **triplet loss**. Each training step produces a triplet `(user, positive_item, negative_item)` and the loss pushes the user embedding closer to the positive item and further from the negative.

**1. Triplet construction — `MovieLensTriplet`**

For every positive `(user, liked_item)` pair (rating ≥ 4), a negative item is sampled randomly from items the user has never rated:

```python
while n in self.user_watched[u]:
    n = np.random.choice(list(self.all_items))
```

This is **in-batch random negative sampling** — cheap and sufficient for a first model.

**2. The loss**

```python
criterion = nn.TripletMarginLoss(margin=0.2)
loss = criterion(u_emb, p_emb, n_emb)
```

Triplet margin loss minimises:

```
max(0,  dist(user, pos) − dist(user, neg) + margin)
```

The `margin=0.2` means the positive must be at least 0.2 closer than the negative before the loss reaches zero — it doesn't just pull the positive in, it simultaneously pushes the negative away.

**3. The two towers**

Both towers share the same structure: embedding lookup → linear projection → L2 normalisation.

```python
def get_user_embs(self, u_ids):
    return F.normalize(self.user_net(self.user_emb(u_ids)), p=2, dim=1)

def get_item_embs(self, i_ids):
    return F.normalize(self.item_net(self.item_emb(i_ids)), p=2, dim=1)
```

L2 normalisation projects every vector onto the unit hypersphere, so the dot product between any user and item vector equals their **cosine similarity** — magnitude cannot inflate scores.

**4. Training loop**

The two towers run together during training (one forward pass for user + positive, a second for the negative item), then triplet loss drives backprop through both:

```python
u_emb, p_emb = model(u, p)
_, n_emb     = model(u, n)
loss = criterion(u_emb, p_emb, n_emb)
loss.backward()
```

**5. Inference**

At eval time the towers decouple — each runs independently over all users/items to build two matrices, then a single matmul scores everything:

```python
scores = user_embs @ item_embs.T   # (n_users, n_items) cosine similarity matrix
_, top_k = torch.topk(scores, k=10, dim=1)
```

**Results after 10 epochs**

| Epoch | Avg Loss |
|---|---|
| 1 | 0.1890 |
| 5 | 0.0629 |
| 10 | 0.0580 |

**Recall@10: 87.15%** — for 87% of users, at least one of their liked films appears in the top-10 predictions (embedding dim=32, RTX 4060).

---

## FAQ — two_tower_NN.ipynb

**Q: What does the full training pipeline look like?**

Each training step consumes one batch of triplets `(user_id, pos_item_id, neg_item_id)`. Both towers share weights across all three forward passes — the item tower runs twice (once for the positive, once for the negative).

```
DATA LOADING
────────────
MovieLens 100k
     │
     ▼
Filter rating ≥ 4  →  55,375 positive (user, item) pairs
     │
     ▼
MovieLensTriplet.__getitem__
  • user_id   ──────────────────────────────────────────────────┐
  • pos_item  (item user liked)                                  │
  • neg_item  (random item user has never seen)  ←─ while-loop  │
     │                                                           │
     ▼                                                           │
DataLoader  (batch_size=64, shuffle=True)                        │
                                                                 │
FORWARD PASS                                                     │
────────────                                                     │
                                                                 │
user_id  ──►  nn.Embedding(n_users, 32)  ──►  nn.Linear(32,32)  ──►  F.normalize  ──►  u_emb  (64×32)
                                                                 │
pos_id   ──►  nn.Embedding(n_items, 32)  ──►  nn.Linear(32,32)  ──►  F.normalize  ──►  p_emb  (64×32)
                                                                 │
neg_id   ──►  nn.Embedding(n_items, 32)  ──►  nn.Linear(32,32)  ──►  F.normalize  ──►  n_emb  (64×32)
          └─────────────── same item_emb + item_net weights ────┘

LOSS & BACKPROP
───────────────

u_emb, p_emb, n_emb
        │
        ▼
TripletMarginLoss(margin=0.2)
  loss = mean( max(0,  dist(u, p)  −  dist(u, n)  +  0.2) )
                         ↑ pull closer    ↑ push apart
        │
        ▼
loss.backward()   ←─ gradients flow back through ALL three passes
        │
        ▼
Adam.step()       ←─ updates Embedding weights + Linear weights for both towers

        ↑
        └── repeat for every batch, every epoch (10 epochs total)


POST-TRAINING  →  BUILD ANN INDEX
──────────────────────────────────

Trained Item Tower  (weights now frozen)
        │
        ▼
Run item tower once over ALL items:
  all_item_ids [0..n_items-1]
        │
        ▼
  nn.Embedding  ──►  nn.Linear  ──►  F.normalize
        │
        ▼
item_embs                        shape: (n_items × 32)
        │
        ├──► Notebook:   kept in GPU memory, queried with torch.topk (exact)
        │
        └──► Production: fed into ANN index
               │
               ├── faiss.IndexFlatIP(32)          ← exact cosine, baseline
               ├── faiss.IndexIVFFlat(...)         ← inverted file, faster
               ├── faiss.IndexHNSWFlat(32, M=32)  ← graph-based, best recall/speed
               └── pgvector / Pinecone / Weaviate  ← managed vector DB
                        │
                        ▼
               index.add(item_embs)   ← one-time build, then served online
               index saved to disk / deployed to vector DB
                        │
                        ▼
               ┌─────────────────────────┐
               │   ANN INDEX  (offline)  │  ← queried by inference pipeline
               │   n_items vectors × 32  │
               └─────────────────────────┘
```

---

**Q: What does the full inference pipeline look like?**

Inference splits into two phases. **Offline inference** runs the item tower once over the full catalogue (frozen weights) to populate the shared embedding space and build the ANN index. **Online inference** runs the user tower at query time.

```
OFFLINE INFERENCE  (frozen weights, run once per retrain cycle)
───────────────────────────────────────────────────────────────

all_item_ids  [0, 1, 2, ..., n_items-1]
        │
        ▼
Item Tower:  (frozen weights)
  nn.Embedding  ──►  nn.Linear  ──►  F.normalize
        │
        ▼
item_embs                        shape: (n_items × 32)   ← items in shared space
        │
        └──► ANN index built (Faiss / pgvector) and deployed


ONLINE  (runs at query time for each user request, microseconds)
────────────────────────────────────────────────────────────────

user_id  (e.g. user 42)
        │
        ▼
User Tower:
  nn.Embedding  ──►  nn.Linear  ──►  F.normalize
        │
        ▼
user_emb                         shape: (1 × 32)

        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  RETRIEVAL  —  find the k nearest item embeddings           │
│                                                             │
│  Notebook (exact):                                          │
│    scores = user_emb @ item_embs.T   (1 × n_items)         │
│    torch.topk(scores, k=10)                                 │
│    → scans ALL items every query, O(n_items × dim)          │
│    → fine for 1,682 films, too slow for millions            │
│                                                             │
│  Production (ANN — Approximate Nearest Neighbour):          │
│    index = faiss.IndexFlatIP(dim)   ← or IVF / HNSW        │
│    index.add(item_embs)             ← built offline once    │
│    _, top_k = index.search(user_emb, k=10)                  │
│    → skips most of the catalogue, O(log n) or sub-linear    │
│    → pgvector, Pinecone, Weaviate all do the same thing     │
│                                                             │
│  Both return the same shape: top-k item indices             │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
top_k_item_ids  [film_3, film_91, film_7, ...]   ← recommendations


EVAL  (Recall@10)
──────────────────

top_k_item_ids  ∩  user_positives[user_id]  →  hit? (1 or 0)
average over all users  →  Recall@10 = 87.15%
```

Key insight: the item tower never runs at query time. Serving a recommendation is one user forward pass (tiny) + an ANN lookup against the pre-cached item index — no matter how large the catalogue, the query time stays sub-linear. The notebook uses exact `topk` because 1,682 films is small enough that brute-force is instant; swap in Faiss/pgvector when the catalogue reaches millions.

---

**Q: Why is the item embedding step called "offline"? Isn't it just a forward pass — i.e. inference?**

Yes, exactly. Running the item tower over all items post-training IS inference — it is a standard forward pass with frozen weights. The word "offline" only describes *when* it runs relative to user queries: before any request arrives, not in response to one. A more precise name is **post-training embedding generation**.

**What training actually produced**

Training optimised both towers jointly so that user and item vectors land in the same shared geometric space — cosine similarity in that space equals "how much this user would like this item". That shared space is the output of training. The item embeddings are not a separate concept; they are the item tower's contribution to that space, frozen once training ends.

```
TRAINING OUTPUT
───────────────

Frozen User Tower weights    Frozen Item Tower weights
         │                            │
         ▼                            ▼
  user_id  ──►  u_emb          item_id  ──►  i_emb
                  └──────────────────────┘
                    shared dot-product space
                    (cosine similarity meaningful here)
```

**Why item embeddings are generated once post-training, not per query**

The item tower is deterministic given frozen weights — `item_tower(id=42)` returns the same vector on every call until the model is retrained. Running it again per user query would recompute identical results:

```
10,000,000 items × 5,000 user queries/sec
  = 50,000,000,000 redundant item tower passes/sec   ← absurd
```

Instead, generate all item embeddings once right after training, build an ANN index from them, and serve millions of user queries against that fixed index:

```
OFFLINE INFERENCE  (frozen weights, once per retrain cycle)
────────────────────────────────────────────────────────────

Item Tower  (frozen)
        │
        ▼  over all item IDs
item_embs  (n_items × dim)   ← items projected into the shared space
        │
        ▼
ANN index built  (Faiss HNSW / pgvector / Pinecone)
  index.add(item_embs)
        │
        ▼
Index deployed  ←──────────────────────────────────────────┐
                                                            │
QUERY TIME  (per user request, real-time)                  │
─────────────────────────────────────────                  │
                                                            │
Frozen User Tower                                           │
        │                                                   │
        ▼  forward pass for this one user                   │
user_emb  (1 × dim)   ← user projected into the shared space
        │                                                   │
        └──► ANN index.search(user_emb, k=10) ─────────────┘
                        │
                        ▼
             Top-k item IDs returned
```

**Item vs user: why the asymmetry**

```
ITEM EMBEDDINGS                        USER EMBEDDINGS
───────────────                        ───────────────
Deterministic until next retrain       Deterministic until next retrain
Catalogue changes rarely               Each query involves exactly 1 user
  (new film added once a week)         Cost of 1 user tower pass: trivial
Cost of ALL items: paid once           → always computed at query time
→ computed once post-training,
  cached in ANN index
```

**What happens when new items are added between retrains?**

New items have no vector in the current index — they are invisible to the retrieval system. This is the **item cold-start problem**. Common mitigations:

- Incremental update: re-run the item tower on just the new IDs and `index.add()` the new vectors without a full retrain
- Content-based fallback: use item metadata features (genre, description) directly for items younger than one retrain cycle
- Retrain more frequently (hourly instead of nightly)

**Why the notebook skips the ANN index**

MovieLens 100k has only 1,682 items. `torch.topk` over 1,682 scores takes microseconds, so there is no practical reason to build an ANN index. The split only matters once the catalogue is large enough that a brute-force scan becomes a throughput or latency bottleneck — typically above ~100k items in a real-time serving context.

---

**Q: What is a Two-Tower model and why is it called that?**

Two separate neural networks (towers) process the user and the item independently and produce embeddings in a shared vector space. They never see each other's raw features during the forward pass — only the final embeddings are compared. The name comes from the two parallel sub-networks standing side by side. This separation is what makes the architecture scalable: at inference time you pre-compute all item embeddings once offline, then a user query is just one tower forward pass + a nearest-neighbour lookup.

---

**Q: Why filter for ratings ≥ 4? What happens to the 1–3 star ratings?**

Ratings ≥ 4 are treated as genuine positive signals — things the user actually liked. Ratings 1–3 are discarded entirely rather than used as explicit negatives. This is intentional: a 3-star rating is ambiguous (lukewarm, not actively disliked), and using it as a hard negative could confuse the model. The negative signal instead comes from random unrated items, which are almost certainly irrelevant to the user.

---

**Q: What exactly is contrastive learning? How does it differ from standard classification?**

Standard classification trains on fixed labels (class 0, class 1). Contrastive learning has no fixed output classes — instead it learns by *comparing* examples. The model is told "these two things should be close, those two should be far apart" and adjusts embeddings accordingly. Triplet loss is one form: given an anchor (user), a positive (liked item), and a negative (irrelevant item), the loss is zero only when `dist(anchor, positive) < dist(anchor, negative) - margin`. The model learns geometry, not class boundaries.

---

**Q: What is negative sampling and why is the while-loop necessary?**

Negative sampling picks a random item the user has *not* interacted with to serve as the "wrong answer" for that training step. The while-loop is needed because a randomly chosen item might accidentally be one the user actually rated highly — using it as a negative would give the model a contradictory signal. The loop resamples until the candidate is genuinely unseen by that user:

```python
n = p
while n in self.user_watched[u]:
    n = np.random.choice(list(self.all_items))
```

For large catalogues this loop almost never iterates more than once, since the odds of hitting a watched item are tiny.

---

**Q: What does `margin=0.2` mean in `TripletMarginLoss`?**

The margin is a safety buffer. The full loss formula is:

```
loss = max(0,  dist(user, pos) − dist(user, neg) + 0.2)
```

Even if the positive is already closer than the negative, the loss is non-zero until the gap exceeds 0.2. Without a margin the model could achieve zero loss by placing all embeddings in a tiny cluster — the margin forces a meaningful separation. Higher margin = stricter requirement = slower convergence but potentially sharper rankings.

---

**Q: Why does the model call `model(u, n)` twice — once for the positive and once for the negative?**

```python
u_emb, p_emb = model(u, p)   # user + positive
_, n_emb     = model(u, n)   # user (discarded) + negative
```

`model.forward()` runs both towers together and returns `(user_emb, item_emb)`. For the negative pass you need the item embedding for `n`, not the user embedding again (it would be identical). The `_` discards the redundant user embedding. An alternative would be to call `model.get_item_embs(n)` directly, which is slightly cleaner.

---

**Q: Why L2-normalise the embeddings? What goes wrong without it?**

Without normalisation, embeddings can grow to arbitrary magnitudes. A user with a very high-norm vector would score highly against *every* item just because of its scale, not because of genuine similarity. L2 normalisation projects every vector onto the unit hypersphere (all norms = 1), so the dot product becomes pure cosine similarity and scores are bounded in [−1, 1]. It also stabilises training — gradients don't explode through high-norm vectors.

---

**Q: What does the `nn.Linear` layer do on top of the embedding? Why not use the raw embedding directly?**

The raw embedding is a lookup — it maps an integer ID to a learned dense vector but applies no transformation. The linear layer (`user_net`, `item_net`) rotates and rescales that vector into a *shared* dot-product space where user and item embeddings are geometrically compatible for comparison. Without it, user embeddings and item embeddings live in separate learned spaces with no guarantee they're aligned. The linear layer is the bridge that makes `user_emb · item_emb` meaningful.

---

**Q: Why does loss stop improving after ~epoch 6 and plateau around 0.058?**

Several reasons:

1. **Easy negatives are exhausted.** Random negatives are usually very easy (the model quickly learns they're wrong). Once all easy negatives yield zero loss, only hard negatives still contribute — and there aren't many of those in random sampling.
2. **Model capacity.** With `dim=32` and a single linear layer per tower, the model has limited expressiveness.
3. **Data saturation.** 55k positives with 10 epochs means the model has seen every interaction multiple times.

To push the loss lower: use hard negative mining (pick negatives the model currently ranks highest), increase `dim`, or add more layers.

---

**Q: How does inference work differently from training?**

During training both towers run together in the same forward pass and gradients flow through both. At inference (`@torch.no_grad()`):

1. Run the item tower once over all item IDs → `(n_items, dim)` matrix, stored offline.
2. For a user query, run the user tower once → `(1, dim)` vector.
3. Score: `user_vec @ item_matrix.T` → `(1, n_items)` — one dot product per item.
4. `torch.topk` returns the top-k item indices.

Step 1 can be pre-computed and cached. Step 2–4 happen at query time in microseconds. This is why two-tower models are used in production: item embeddings are static until retrained, and user scoring is a single matrix multiply.

---

**Q: What does Recall@10 = 87.15% actually measure?**

For each user, take their set of positively-rated items (ground truth) and the model's top-10 predicted items. A "hit" is when the intersection is non-empty — at least one liked item appears in the top 10. Recall@10 is the fraction of users who get at least one hit:

```python
hits = sum(1 for u in range(n_users)
           if user_positives.get(u, set()) & set(top_k[u]))
return hits / n_users
```

87.15% is strong for this setup. Note this is a lenient metric (one hit suffices) — Precision@10 or NDCG@10 would give a stricter picture of ranking quality.

---

**Q: What are the main limitations of this implementation?**

| Limitation | Impact | Fix |
|---|---|---|
| Random negative sampling | Easy negatives → slow convergence, inflated Recall | Hard negative mining |
| No side features | Can't cold-start new users/items with no history | Add content features (genre, user demographics) to the towers |
| Single linear layer | Limited capacity for complex preference patterns | Deeper MLP towers |
| `dim=32` | Small embedding space | Try 64–256 |
| No train/val/test split | Recall@10 is measured on training data | Temporal split (train on earlier ratings, eval on later) |
| Leaky ground truth in eval | `positives_df` used for both training and eval | Hold out a test set before training |

---

**Q: How would you take this to production?**

1. Pre-compute all item embeddings and store them in a vector database (e.g. pgvector, Pinecone, Faiss).
2. At request time, run only the user tower → query the vector DB for nearest neighbours.
3. Retrain periodically (daily/weekly) as new interaction data arrives.
4. For new users with no history, fall back to popularity-based or content-based recommendations until enough interactions accumulate.

---
---

## FAQ — minerU_2.5.ipynb

**Q: What is MinerU 2.5?**

MinerU is an open-source document intelligence toolkit from OpenDataLab. Version 2.5 adds a vision-language (VL) pipeline powered by a fine-tuned Qwen2-VL model. It extracts structured content from document images — titles, headers, body text, tables, figures — and returns them as typed blocks with normalised bounding boxes. It works on scanned PDFs, photos of documents, receipts, ID cards, and technical reports.

---

**Q: What model does it actually use under the hood?**

`opendatalab/MinerU2.5-2509-1.2B` — a 1.2 billion parameter vision-language model fine-tuned from Qwen2-VL on document understanding tasks. "2509" in the name is the training checkpoint date (September 2025). It's loaded via HuggingFace Transformers:

```python
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "opendatalab/MinerU2.5-2509-1.2B", dtype="auto", device_map="auto"
)
```

`device_map="auto"` places the model on GPU if available, CPU otherwise. `dtype="auto"` lets HuggingFace pick the best precision (bfloat16 on Ampere+ GPUs).

---

**Q: What does `client.two_step_extract()` do?**

It runs a two-stage pipeline on a single document image:

1. **Layout detection** — identifies where each content block is on the page and classifies its type (title, text, table, figure, etc.), returning bounding boxes.
2. **Content extraction** — for each detected block, runs the VL model to read out the actual text or HTML (for tables).

The result is a list of dicts, one per block:

```python
[
  {"type": "title",  "content": "NVIDIA Corporation",  "bbox": [0.1, 0.05, 0.9, 0.12]},
  {"type": "text",   "content": "Revenue for Q3...",   "bbox": [0.1, 0.15, 0.9, 0.30]},
  {"type": "table",  "content": "<table>...</table>",  "bbox": [0.1, 0.35, 0.9, 0.60]},
  ...
]
```

---

**Q: What block types does MinerU detect and what color is each in the visualisation?**

| Block type | Color | Typical content |
|---|---|---|
| `title` | Dark Red | Document or section title |
| `header` | Red-Orange | Running header / page header |
| `text` | Blue | Body paragraphs |
| `list` | Purple | Bullet or numbered lists |
| `table` | Green | Tabular data (returned as HTML) |
| `table_caption` | Teal | Caption text above/below a table |
| `figure` | Orange | Images, charts, diagrams |
| `default` | Dark Grey | Anything unrecognised |

---

**Q: What are "normalised bounding boxes" and how are they converted to pixels?**

The model returns bounding boxes as `[x1, y1, x2, y2]` with values in `[0, 1]` relative to the image dimensions. This makes them resolution-independent. `display_extraction` converts to pixel coordinates for drawing:

```python
x1, y1, x2, y2 = bbox
rect_coords = [x1 * width, y1 * height, x2 * width, y2 * height]
draw.rectangle(rect_coords, outline=color, width=3)
```

A bbox of `[0.1, 0.05, 0.9, 0.12]` on a 1000×800 image becomes the rectangle `[100, 40, 900, 96]` in pixels.

---

**Q: What documents were tested in the notebook?**

Four document types from the `ocr-documents.zip` archive:

| File | Document type |
|---|---|
| `nvidia-first-page.jpg` | Corporate annual report cover page |
| `nvidia-inner-page.jpg` | Dense financial/technical inner page with tables |
| `receipt.jpg` | Retail receipt |
| `id-card.png` | ID card |

This range tests MinerU across very different layouts — structured corporate docs, semi-structured receipts, and identity documents.

---

**Q: What is `MinerUClient` and why use it instead of calling the model directly?**

`MinerUClient` from `mineru-vl-utils` is a thin wrapper that handles the two-step pipeline (layout detection → content extraction) and manages prompt construction for the Qwen2-VL model. Calling the model directly would require you to manually craft the VL prompt, parse the raw output, and convert coordinates — `MinerUClient` abstracts all of that into a single `two_step_extract(image)` call.

---

**Q: What is `magic-pdf.json` and why is it configured?**

MinerU's CLI tool (`mineru`) reads `~/.config/magic-pdf.json` for runtime settings. The notebook sets `device-mode: "cuda"` and points `models-dir` at the HuggingFace cache path so the CLI knows where the downloaded model lives and runs it on GPU:

```python
config_data = {
    "device-mode": "cuda",
    "models-dir": "C:\\Users\\rohan\\.cache\\huggingface\\hub\\models--opendatalab--MinerU2.5-2509-1.2B"
}
```

This only matters when using the CLI (`mineru -p input.pdf`). The Python API (`MinerUClient`) doesn't read this file.

---

**Q: How do I run MinerU on a PDF from the command line?**

```bash
# GPU
CUDA_VISIBLE_DEVICES=0 mineru -p /path/to/input.pdf -o /path/to/output/

# CPU only
mineru -p /path/to/input.pdf -o /path/to/output/ --device cpu
```

The output directory will contain Markdown, JSON (structured blocks), and images for each extracted figure/table. The `CUDA_VISIBLE_DEVICES=0` flag selects the first GPU.

---

**Q: How does the side-by-side display work?**

`display_extraction` produces an inline HTML page rendered directly in the Jupyter output cell:

- **Left pane** — the original image with coloured bounding boxes and type labels drawn on it using `PIL.ImageDraw`, then base64-encoded and embedded as a `data:image/png` URI (no file I/O needed).
- **Right pane** — the extracted text rendered as HTML: titles become `<h2>`, headers `<h4>`, tables are injected as raw `<table>` HTML, everything else is `<p>`. The pane has `overflow-y: auto` so long documents scroll.

---

**Q: Does MinerU work without a GPU?**

Yes, but slowly. The 1.2B parameter model runs on CPU; expect seconds to minutes per page depending on hardware. The notebook checks GPU availability and `magic-pdf.json` sets `device-mode: "cuda"`. For CPU-only use set `device-mode: "cpu"` in the config and remove `CUDA_VISIBLE_DEVICES` from the CLI call. The Python API respects `device_map="auto"` which falls back to CPU automatically.

---

**Q: Why does the notebook call `!pip install -U "mineru[all]"` twice (cell 2 and cell 23)?**

Cell 23 is a leftover re-run of the install cell, likely from debugging a missing dependency mid-session. It's harmless (pip is idempotent) but redundant — it can be deleted.

---

**Q: How do I use MinerU on my own documents?**

```python
from mineru_vl_utils import MinerUClient
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "opendatalab/MinerU2.5-2509-1.2B", dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    "opendatalab/MinerU2.5-2509-1.2B", use_fast=True
)
client = MinerUClient(backend="transformers", model=model, processor=processor)

blocks = client.two_step_extract(Image.open("your_document.jpg"))
for b in blocks:
    print(b["type"], "|", b["content"][:80])
```

For multi-page PDFs, convert each page to an image first (e.g. with `pdf2image`) and call `two_step_extract` per page.
