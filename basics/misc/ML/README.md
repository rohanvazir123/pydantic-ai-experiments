# ML Notebooks

## Table of Contents

**two_tower_NN.ipynb**
- [Overview & Contrastive Learning](#two_towernnipynb)
- [Q1. Training pipeline diagram](#q1-what-does-the-full-training-pipeline-look-like)
- [Q2. Inference pipeline diagram](#q2-what-does-the-full-inference-pipeline-look-like)
- [Q3. Why do we need offline inference?](#q3-why-do-we-need-offline-inference-and-why-is-a-forward-pass-called-inference-at-all)
- [Q4. Where are embeddings stored? Are they the trained weights?](#q4-where-are-user-and-item-embeddings-stored-in-the-model-are-they-the-trained-weights)
- [Q5. What is a Two-Tower model?](#q5-what-is-a-two-tower-model-and-why-is-it-called-that)
- [Q6. Why filter ratings ≥ 4?](#q6-why-filter-for-ratings--4-what-happens-to-the-13-star-ratings)
- [Q7. What is contrastive learning?](#q7-what-exactly-is-contrastive-learning-how-does-it-differ-from-standard-classification)
- [Q8. What is negative sampling?](#q8-what-is-negative-sampling-and-why-is-the-while-loop-necessary)
- [Q9. What does margin=0.2 mean?](#q9-what-does-margin02-mean-in-tripletmarginloss)
- [Q10. Why call model(u, n) twice?](#q10-why-does-the-model-call-modelu-n-twice--once-for-the-positive-and-once-for-the-negative)
- [Q11. Why L2-normalise embeddings?](#q11-why-l2-normalise-the-embeddings-what-goes-wrong-without-it)
- [Q12. What does nn.Linear do on top of the embedding?](#q12-what-does-nnlinear-do-on-top-of-the-embedding-why-not-use-the-raw-embedding-directly)
- [Q13. Why does loss plateau at ~0.058?](#q13-why-does-loss-stop-improving-after-epoch-6-and-plateau-around-0058)
- [Q14. How does inference differ from training?](#q14-how-does-inference-work-differently-from-training)
- [Q15. What does Recall@10 measure?](#q15-what-does-recall10--8715-actually-measure)
- [Q16. Main limitations](#q16-what-are-the-main-limitations-of-this-implementation)
- [Q17. Which models can I use?](#q17-which-models-can-i-swap-in-or-experiment-with-in-this-notebook)
- [Q18. What datasets simulate production?](#q18-what-other-datasets-can-i-use-to-simulate-production-conditions)
- [Q19. How to take this to production?](#q19-how-would-you-take-this-to-production)
- [Q20. How does final ranking work? LightGBM / XGBoost?](#q20-how-does-final-ranking-work-in-production-do-systems-use-lightgbm-or-xgboost)

**minerU_2.5.ipynb**
- [Q1. What is MinerU 2.5?](#q1-what-is-mineru-25)
- [Q2. What model does it use?](#q2-what-model-does-it-actually-use-under-the-hood)
- [Q3. What does two_step_extract() do?](#q3-what-does-clienttwo_step_extract-do)
- [Q4. What block types does it detect?](#q4-what-block-types-does-mineru-detect-and-what-color-is-each-in-the-visualisation)
- [Q5. What are normalised bounding boxes?](#q5-what-are-normalised-bounding-boxes-and-how-are-they-converted-to-pixels)
- [Q6. What documents were tested?](#q6-what-documents-were-tested-in-the-notebook)
- [Q7. What is MinerUClient?](#q7-what-is-mineruclient-and-why-use-it-instead-of-calling-the-model-directly)
- [Q8. What is magic-pdf.json?](#q8-what-is-magic-pdfjson-and-why-is-it-configured)
- [Q9. How to run on a PDF from CLI?](#q9-how-do-i-run-mineru-on-a-pdf-from-the-command-line)
- [Q10. How does the side-by-side display work?](#q10-how-does-the-side-by-side-display-work)
- [Q11. Does it work without a GPU?](#q11-does-mineru-work-without-a-gpu)
- [Q12. Why is pip install called twice?](#q12-why-does-the-notebook-call-pip-install--u-mineruall-twice-cell-2-and-cell-23)
- [Q13. How to use on my own documents?](#q13-how-do-i-use-mineru-on-my-own-documents)

---

## two_tower_NN.ipynb

A recommender system trained on MovieLens 100k using a Two-Tower neural network and contrastive learning. The entire model — embeddings, linear layers, loss, and training loop — is written in **PyTorch** (`torch`, `torch.nn`, `torch.nn.functional`).

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

> **Note:** Average loss reached ~5.8% in just 10 epochs. Fast convergence here is largely a function of data quality — MovieLens 100k is a clean, well-curated dataset with consistent explicit ratings. Noisy or sparse real-world interaction data (clicks, dwell time, implicit signals) typically requires more epochs and harder negative mining to reach comparable loss.

---

## FAQ — two_tower_NN.ipynb

### Q1: What does the full training pipeline look like?

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

### Q2: What does the full inference pipeline look like?

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


ONLINE INFERENCE  (runs at query time for each user request, microseconds)
───────────────────────────────────────────────────────────────────────────

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
│  Both return k item indices ranked by cosine similarity     │
│  score (highest first) — rank 1 = most similar to user     │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
top_k_item_ids  [film_3, film_91, film_7, ...]
  ranked by cosine similarity score (descending)
  film_3 = highest score = strongest match for this user


EVAL  (Recall@10)
──────────────────

top_k_item_ids  ∩  user_positives[user_id]  →  hit? (1 or 0)
average over all users  →  Recall@10 = 87.15%
```

Key insight: the item tower never runs at query time. Serving a recommendation is one user forward pass (tiny) + an ANN lookup against the pre-cached item index — no matter how large the catalogue, the query time stays sub-linear. The notebook uses exact `topk` because 1,682 films is small enough that brute-force is instant; swap in Faiss/pgvector when the catalogue reaches millions.

---

### Q3: Why do we need offline inference? And why is a forward pass called "inference" at all?

Running the item tower over all items post-training IS inference — it is a standard forward pass with frozen weights. The word "offline" only describes *when* it runs relative to user queries: before any request arrives, not in response to one.

**Every time training runs, offline inference must run too**

Offline inference is not a one-time setup step — it is a mandatory post-step after **every training run**. When training completes, the model weights change. That means all previously computed item embeddings are stale — they were produced by the old weights and no longer reflect the new shared space. The ANN index built from them is therefore also stale and must be rebuilt from scratch:

```
Training run completes  (new weights)
        │
        ▼
Old item embeddings  ──►  STALE, discard
Old ANN index        ──►  STALE, discard
        │
        ▼
Offline inference: run item tower (new weights) over all item IDs
        │
        ▼
New item embeddings  (aligned with new shared space)
        │
        ▼
Rebuild ANN index from new embeddings
        │
        ▼
Deploy new index  →  online inference resumes against fresh index
```

This is why retraining pipelines always include an embedding generation + index rebuild step before the new model goes live. Skipping it means users are queried against an index that doesn't match the new model's geometry — scores would be meaningless.

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
ONLINE INFERENCE  (per user request, real-time)            │
────────────────────────────────────────────────           │
                                                            │
Frozen User Tower                                           │
        │                                                   │
        ▼  forward pass for this one user                   │
user_emb  (1 × dim)   ← user projected into the shared space
        │                                                   │
        └──► ANN index.search(user_emb, k=10) ─────────────┘
                        │
                        ▼
             Top-k item IDs returned,
             ranked by cosine similarity score (descending)
             rank 1 = highest score = strongest match
```

**Item vs user: why the asymmetry**

```
ITEM EMBEDDINGS                        USER EMBEDDINGS
───────────────                        ───────────────
Deterministic until next retrain       Deterministic until next retrain
Catalogue changes rarely               Each query involves exactly 1 user
  (new film added once a week)         Cost of 1 user tower pass: trivial
Cost of ALL items: paid once           → always computed at query time
→ offline inference, cached in ANN
```

**What happens when new items are added between retrains?**

New items have no vector in the current index — they are invisible to the retrieval system. This is the **item cold-start problem**. Common mitigations:

- Incremental update: re-run the item tower on just the new IDs and `index.add()` the new vectors without a full retrain
- Content-based fallback: use item metadata features (genre, description) directly for items younger than one retrain cycle
- Retrain more frequently (hourly instead of nightly)

**Why the notebook skips the ANN index**

MovieLens 100k has only 1,682 items. `torch.topk` over 1,682 scores takes microseconds, so there is no practical reason to build an ANN index. The split only matters once the catalogue is large enough that a brute-force scan becomes a throughput or latency bottleneck — typically above ~100k items in a real-time serving context.

---

### Q4: Where are user and item embeddings stored in the model? Are they the trained weights?

Yes — the embeddings **are** the weights. `nn.Embedding` is nothing more than a learnable matrix stored as a parameter in the model. Looking up an embedding is just indexing a row from that matrix.

```python
self.user_emb = nn.Embedding(n_users, dim)  # weight matrix: (n_users × 32)
self.item_emb = nn.Embedding(n_items, dim)  # weight matrix: (n_items × 32)
```

These are initialised randomly and updated by backprop on every training step, exactly like any other weight in the network. The full set of trained parameters in `TwoTower` is:

```
PARAMETER              SHAPE           WHAT IT STORES
──────────────────────────────────────────────────────────────
user_emb.weight        (n_users × 32)  one row = one user's raw embedding
item_emb.weight        (n_items × 32)  one row = one item's raw embedding
user_net.weight        (32 × 32)       linear projection for user tower
user_net.bias          (32,)
item_net.weight        (32 × 32)       linear projection for item tower
item_net.bias          (32,)
```

**What the embedding lookup actually does**

```python
# This line:
self.user_emb(u_ids)

# Is equivalent to:
self.user_emb.weight[u_ids]   # just row indexing — no multiplication
```

`u_ids` is a LongTensor of integer user IDs. The embedding layer uses them as row indices to fetch the corresponding rows from `user_emb.weight`. The gradient flows back into those specific rows during backprop, updating only the rows that were looked up in that batch.

**So where do the final embeddings (used for ANN) come from?**

The raw embedding rows from `user_emb.weight` / `item_emb.weight` are **not** what gets stored in the ANN index. They are not similarity scores — they are raw learned representations that haven't yet been aligned or scaled. They must pass through two more steps before they are meaningful for comparison:

```
user_emb.weight[user_id]
        │
        │  Raw weight row — a point in 32-D space.
        │  NOT a similarity score. NOT comparable to item vectors yet.
        │  User tower and item tower have separate embedding matrices,
        │  so their raw rows live in different learned spaces with no
        │  guarantee of geometric alignment.
        │
        ▼
STEP 1 — LINEAR PROJECTION:  user_net(x) = W @ x + b
        │
        │  W is (32 × 32), b is (32,) — both learned during training.
        │  This is a rotation + scaling of the raw embedding vector.
        │  Its purpose is to project the user vector into the SAME
        │  geometric space as the item vectors, so that a dot product
        │  between them is meaningful.
        │
        │  Training optimises W (and the item tower's equivalent) so
        │  that after projection, liked (user, item) pairs end up close
        │  and disliked pairs end up far — this is the shared space.
        │
        │  Output: still (32,), but now in the shared dot-product space.
        │  Still NOT a similarity score — norms are unconstrained.
        │
        ▼
STEP 2 — L2 NORMALISATION:  F.normalize(x, p=2, dim=1)
        │
        │  Divides the vector by its own L2 norm:
        │    x_norm = x / ||x||₂
        │
        │  This projects every vector onto the UNIT HYPERSPHERE —
        │  all vectors now have norm = 1, regardless of magnitude.
        │
        │  Why this matters:
        │    dot(u, i) = ||u|| * ||i|| * cos(θ)
        │    if ||u|| = ||i|| = 1:
        │    dot(u, i) = cos(θ)   ← pure cosine similarity
        │
        │  Without normalisation, a high-norm vector scores highly
        │  against everything regardless of direction — magnitude
        │  would dominate over actual similarity.
        │
        ▼
unit-norm vector  (32,)   ← THIS goes into the ANN index

        Are these similarity scores? NO — not yet.
        They are REPRESENTATIONS (points on the unit hypersphere).
        The similarity score is computed LATER at query time:

        score = dot(user_vec, item_vec) = cos(θ)   ∈ [-1, 1]

        score =  1.0  →  vectors point in exactly the same direction
                          (perfect match)
        score =  0.0  →  vectors are orthogonal (unrelated)
        score = -1.0  →  vectors point in opposite directions
                          (strong mismatch)

        torch.topk / ANN index ranks items by this score descending.
        Rank 1 = highest cosine similarity = strongest predicted match.
```

**Summary of what each value is**

```
VALUE                        IS IT A SIMILARITY SCORE?
─────────────────────────────────────────────────────────────
user_emb.weight[user_id]     No  — raw learned representation, unaligned
after user_net(x)            No  — aligned to shared space, unconstrained norm
after F.normalize(x)         No  — unit-norm representation (stored in ANN)
dot(user_vec, item_vec)      YES — cosine similarity score ∈ [-1, 1]
torch.topk result            YES — top-k items ranked by cosine similarity
```

The weights themselves stay inside the model file and are never directly stored in the ANN index.

**Do we compute dot products for every user and every item during offline inference?**

No — offline inference computes **zero dot products**. It only runs the item tower forward pass. Users are not involved at all.

```
OFFLINE INFERENCE  (item tower only, no users, no dot products)
────────────────────────────────────────────────────────────────

item_id=0  ──►  item tower  ──►  unit-norm vec  (32,)  ┐
item_id=1  ──►  item tower  ──►  unit-norm vec  (32,)  │
item_id=2  ──►  item tower  ──►  unit-norm vec  (32,)  │  stacked into
   ...                                                  │  item_embs
item_id=N  ──►  item tower  ──►  unit-norm vec  (32,)  ┘  (N × 32)
                                                        │
                                                        ▼
                                               ANN index.add(item_embs)
                                               — organise into search
                                                 structure, no dot
                                                 products computed here
                                               — index saved to disk

dot products: 0
user tower:   never called
```

Dot products only happen later, at query time, when a specific user's embedding is compared against the index:

```
WHERE DOT PRODUCTS ACTUALLY HAPPEN
───────────────────────────────────────────────────────────────────────────────
OFFLINE INFERENCE    item tower fwd pass over all items    0 dot products
ANN index build      organise item vectors into graph/     item-item dot products
                     clusters internally — see note below  (NOT user-item)
───────────────────────────────────────────────────────────────────────────────
ONLINE (brute-force) user_emb @ item_embs.T                n_items per query
ONLINE (ANN/HNSW)    graph traversal, ~200-500 nodes        ≪ n_items per query
NOTEBOOK EVAL        user_embs @ item_embs.T                n_users × n_items
```

The separation is the entire point of the architecture: item vectors are computed once offline, user vectors are computed per query, and the ANN index makes sure the dot products at query time are sub-linear rather than exhaustive.

**But how does the ANN index get updated without recomputing similarity scores?**

The ANN index does not store similarity scores. It stores the raw item embedding vectors and organises them into a spatial search structure. Scores are never pre-computed or stored anywhere — they are computed on the fly at query time against whoever is asking.

```
WHAT THE ANN INDEX ACTUALLY CONTAINS
──────────────────────────────────────────────────────────────────
NOT this:
  user_1 → [film_3: 0.91, film_7: 0.87, film_91: 0.83, ...]   ✗
  user_2 → [film_7: 0.95, film_1: 0.72, ...]                   ✗

YES this:
  A spatial data structure over item vectors only.
  No users. No scores. Just item points in 32-D space.

  film_3  →  [0.12, -0.43, 0.71, ...]   (32,) unit-norm vector
  film_7  →  [0.55,  0.22, -0.31, ...]  (32,) unit-norm vector
  film_91 →  [-0.08, 0.61,  0.44, ...]  (32,) unit-norm vector
  ...
```

When the model is retrained, the item vectors change (new weights → new geometry). To update the ANN index you rebuild it from the new item vectors. This does involve dot product computations internally — but they are **item-item**, not user-item:

```
REBUILDING THE ANN INDEX AFTER RETRAIN
────────────────────────────────────────
Old item_embs  (stale, from old weights)  →  discard old index
        │
        ▼
Offline inference: item tower (new weights) over all item IDs
        │
        ▼
New item_embs  (N × 32)   ← new vectors in new geometry
        │
        ▼
index = faiss.IndexHNSWFlat(32, M=32)
index.add(new_item_embs)   ← load new vectors into spatial structure
        │
        ▼
New index deployed  ← scores not stored; no users involved
```

**What are those internal dot products during index build?**

They are item-item distance computations used to organise the spatial structure. No user is involved. The nature depends on the index type:

```
HNSW (Hierarchical Navigable Small World)
  index.add(item_embs):
    For each new item being inserted, compute dot products against
    its M nearest neighbours to wire it into the graph.
    Cost: O(N × M × log N) item-item dot products total.
    M=32 is typical — each node connects to 32 others.
    Result: a multi-layer proximity graph over item vectors.

IVF (Inverted File Index)
  index.train(item_embs):
    Run k-means on all item vectors to find K centroids.
    Cost: O(N × K × iterations) item-centroid dot products.
    Result: K cluster centroids.
  index.add(item_embs):
    Assign each item to its nearest centroid.
    Cost: O(N × K) dot products.
    Result: items bucketed into K inverted lists.
```

Both are one-time costs at index build time — paid once per retrain cycle, not per query. The distinction that matters:

```
INDEX BUILD  →  item-item dot products   (organise spatial structure)
QUERY TIME   →  user-item dot products   (find nearest items for this user)
```

Users never appear during index build. Items never need to know about users to form a good spatial structure — they just need to know which other items are nearby in the embedding space.

The scores only appear when a user query arrives:

```
QUERY TIME  (scores computed here, not before)
───────────────────────────────────────────────
user_emb (1 × 32)  arrives
        │
        ▼
index.search(user_emb, k=10)
  → HNSW traverses its graph, computing dot products against
    ~200-500 nearby item vectors (not all N)
  → returns the k item IDs with the highest dot product scores
        │
        ▼
Scores are returned to the caller — they are not stored in the index
Next query for a different user repeats the same process from scratch
```

So to directly answer: the ANN index never needs similarity scores to be updated. You update it by replacing the item vectors inside it. The scores are ephemeral — computed per query, returned, then gone.

---

### Q5: What is a Two-Tower model and why is it called that?

Two separate neural networks (towers) process the user and the item independently and produce embeddings in a shared vector space. They never see each other's raw features during the forward pass — only the final embeddings are compared. The name comes from the two parallel sub-networks standing side by side. This separation is what makes the architecture scalable: at inference time you pre-compute all item embeddings once offline, then a user query is just one tower forward pass + a nearest-neighbour lookup.

---

### Q6: Why filter for ratings ≥ 4? What happens to the 1–3 star ratings?

Ratings ≥ 4 are treated as genuine positive signals — things the user actually liked. Ratings 1–3 are discarded entirely rather than used as explicit negatives. This is intentional: a 3-star rating is ambiguous (lukewarm, not actively disliked), and using it as a hard negative could confuse the model. The negative signal instead comes from random unrated items, which are almost certainly irrelevant to the user.

---

### Q7: What exactly is contrastive learning? How does it differ from standard classification?

Standard classification trains on fixed labels (class 0, class 1). Contrastive learning has no fixed output classes — instead it learns by *comparing* examples. The model is told "these two things should be close, those two should be far apart" and adjusts embeddings accordingly. Triplet loss is one form: given an anchor (user), a positive (liked item), and a negative (irrelevant item), the loss is zero only when `dist(anchor, positive) < dist(anchor, negative) - margin`. The model learns geometry, not class boundaries.

---

### Q8: What is negative sampling and why is the while-loop necessary?

Negative sampling picks a random item the user has *not* interacted with to serve as the "wrong answer" for that training step. The while-loop is needed because a randomly chosen item might accidentally be one the user actually rated highly — using it as a negative would give the model a contradictory signal. The loop resamples until the candidate is genuinely unseen by that user:

```python
n = p
while n in self.user_watched[u]:
    n = np.random.choice(list(self.all_items))
```

For large catalogues this loop almost never iterates more than once, since the odds of hitting a watched item are tiny.

---

### Q9: What does `margin=0.2` mean in `TripletMarginLoss`?

The margin is a safety buffer. The full loss formula is:

```
loss = max(0,  dist(user, pos) − dist(user, neg) + 0.2)
```

Even if the positive is already closer than the negative, the loss is non-zero until the gap exceeds 0.2. Without a margin the model could achieve zero loss by placing all embeddings in a tiny cluster — the margin forces a meaningful separation. Higher margin = stricter requirement = slower convergence but potentially sharper rankings.

---

### Q10: Why does the model call `model(u, n)` twice — once for the positive and once for the negative?

```python
u_emb, p_emb = model(u, p)   # user + positive
_, n_emb     = model(u, n)   # user (discarded) + negative
```

`model.forward()` runs both towers together and returns `(user_emb, item_emb)`. For the negative pass you need the item embedding for `n`, not the user embedding again (it would be identical). The `_` discards the redundant user embedding. An alternative would be to call `model.get_item_embs(n)` directly, which is slightly cleaner.

---

### Q11: Why L2-normalise the embeddings? What goes wrong without it?

Without normalisation, embeddings can grow to arbitrary magnitudes. A user with a very high-norm vector would score highly against *every* item just because of its scale, not because of genuine similarity. L2 normalisation projects every vector onto the unit hypersphere (all norms = 1), so the dot product becomes pure cosine similarity and scores are bounded in [−1, 1]. It also stabilises training — gradients don't explode through high-norm vectors.

---

### Q12: What does the `nn.Linear` layer do on top of the embedding? Why not use the raw embedding directly?

The raw embedding is a lookup — it maps an integer ID to a learned dense vector but applies no transformation. The linear layer (`user_net`, `item_net`) rotates and rescales that vector into a *shared* dot-product space where user and item embeddings are geometrically compatible for comparison. Without it, user embeddings and item embeddings live in separate learned spaces with no guarantee they're aligned. The linear layer is the bridge that makes `user_emb · item_emb` meaningful.

---

### Q13: Why does loss stop improving after ~epoch 6 and plateau around 0.058?

Reaching ~5.8% loss in only 10 epochs is fast — the main reason is **data quality**. MovieLens 100k is a clean, explicitly rated dataset: every positive is a deliberate 4–5 star rating with no ambiguity, and negatives are genuinely unseen items. That clean signal lets the model converge quickly.

The plateau itself has three causes:

1. **Easy negatives are exhausted.** Random negatives are usually very easy (the model quickly learns they're wrong). Once all easy negatives yield zero loss, only hard negatives still contribute — and there aren't many of those in random sampling.
2. **Model capacity.** With `dim=32` and a single linear layer per tower, the model has limited expressiveness.
3. **Data saturation.** 55k positives with 10 epochs means the model has seen every interaction multiple times.

On noisier real-world data (implicit signals like clicks or dwell time), the same architecture would take more epochs and likely plateau at a higher loss — the clean explicit ratings are doing a lot of the work here.

To push the loss lower: use hard negative mining (pick negatives the model currently ranks highest), increase `dim`, or add more layers.

---

### Q14: How does inference work differently from training?

During training both towers run together in the same forward pass and gradients flow through both. At inference (`@torch.no_grad()`):

1. Run the item tower once over all item IDs → `(n_items, dim)` matrix, stored offline.
2. For a user query, run the user tower once → `(1, dim)` vector.
3. Score: `user_vec @ item_matrix.T` → `(1, n_items)` — one dot product per item.
4. `torch.topk` returns the top-k item indices.

Step 1 can be pre-computed and cached. Step 2–4 happen at query time in microseconds. This is why two-tower models are used in production: item embeddings are static until retrained, and user scoring is a single matrix multiply.

---

### Q15: What does Recall@10 = 87.15% actually measure?

For each user, take their set of positively-rated items (ground truth) and the model's top-10 predicted items. A "hit" is when the intersection is non-empty — at least one liked item appears in the top 10. Recall@10 is the fraction of users who get at least one hit:

```python
hits = sum(1 for u in range(n_users)
           if user_positives.get(u, set()) & set(top_k[u]))
return hits / n_users
```

87.15% is strong for this setup. Note this is a lenient metric (one hit suffices) — Precision@10 or NDCG@10 would give a stricter picture of ranking quality.

---

### Q16: What are the main limitations of this implementation?

| Limitation | Impact | Fix |
|---|---|---|
| Random negative sampling | Easy negatives → slow convergence, inflated Recall | Hard negative mining |
| No side features | Can't cold-start new users/items with no history | Add content features (genre, user demographics) to the towers |
| Single linear layer | Limited capacity for complex preference patterns | Deeper MLP towers |
| `dim=32` | Small embedding space | Try 64–256 |
| No train/val/test split | Recall@10 is measured on training data | Temporal split (train on earlier ratings, eval on later) |
| Leaky ground truth in eval | `positives_df` used for both training and eval | Hold out a test set before training |

---

### Q17: Which models can I swap in or experiment with in this notebook?

The current model is the simplest viable Two-Tower: one embedding + one linear per tower. Everything below plugs into the same training loop and `TripletMarginLoss` — only the `TwoTower` class changes.

**Drop-in tower upgrades (same interface, more capacity)**

```python
# Current — 1 linear layer
self.user_net = nn.Linear(dim, dim)

# Deeper MLP tower — more expressive, same input/output shape
self.user_net = nn.Sequential(
    nn.Linear(dim, dim * 2),
    nn.ReLU(),
    nn.Linear(dim * 2, dim),
)

# With dropout for regularisation
self.user_net = nn.Sequential(
    nn.Linear(dim, dim * 2),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(dim * 2, dim),
)
```

**Alternative architectures (require more structural changes)**

| Model | What changes | Why try it |
|---|---|---|
| **Matrix Factorisation** | Remove linear layers entirely — just `Embedding + F.normalize` | Simplest baseline; shows how much the linear layer helps |
| **Neural CF (NCF)** | Concatenate user + item embeddings, pass through shared MLP for binary output | Uses BCE loss instead of triplet; classic paper benchmark |
| **LightGCN** | Replace embedding lookup with graph convolution over user-item interaction graph | State-of-the-art on sparse data; needs `torch_geometric` |
| **SASRec** | User tower becomes a Transformer over the user's interaction history | Models sequential behaviour — "next item" prediction |
| **Wide & Deep** | Add raw feature inputs (genre, user age) alongside ID embeddings | Tests whether side features improve over ID-only |

**Larger embedding dim**

The current `dim=32` is small. Try `64`, `128`, or `256` — loss and Recall@10 should both improve at the cost of more memory and a larger ANN index.

**Loss function alternatives**

```python
# Current
nn.TripletMarginLoss(margin=0.2)

# BPR loss (Bayesian Personalised Ranking) — equivalent idea, different formulation
loss = -torch.log(torch.sigmoid(pos_score - neg_score)).mean()

# InfoNCE / in-batch negatives — treat all other items in the batch as negatives
# more efficient: one forward pass, N-1 negatives per sample for free
```

---

### Q18: What other datasets can I use to simulate production conditions?

MovieLens 100k is too clean and small to surface real production problems. These datasets add the noise, scale, and implicit signal patterns you'd face in production:

**Larger MovieLens variants (easiest starting point)**

| Dataset | Users | Items | Ratings | Get it |
|---|---|---|---|---|
| MovieLens 1M | 6,040 | 3,706 | 1M | `grouplens.org/datasets/movielens/1m/` |
| MovieLens 10M | 72,000 | 10,681 | 10M | `grouplens.org/datasets/movielens/10m/` |
| MovieLens 20M | 138,000 | 27,000 | 20M | `grouplens.org/datasets/movielens/20m/` |

Same schema as 100k — just change the URL. Good for benchmarking how Recall@10 and training time scale.

**Implicit feedback datasets (simulate click/view signals, no explicit ratings)**

| Dataset | Domain | Scale | Why useful |
|---|---|---|---|
| **Amazon Reviews** | E-commerce (books, electronics, etc.) | Millions | Sparse, noisy, cold-start heavy — closest to real retail |
| **Yelp Open Dataset** | Restaurants / local business | 7M reviews | Geographic signal, text reviews available for content features |
| **Steam Games** | Video games | 200k users, 11k games | Implicit playtime signal, strong cold-start problem |
| **LastFM** | Music listening | 1,000 users, 170k artists | Play counts as implicit signal, good for sequential models |
| **Pinterest** | Image pinning | 55M interactions | Implicit only, very sparse — tests ANN at scale |

**RecSys challenge datasets (production-grade)**

| Dataset | Domain | What makes it hard |
|---|---|---|
| **Criteo** | Display advertising CTR | 45M samples, extreme sparsity, 1-bit implicit signal |
| **Netflix Prize** | Movies | 100M ratings, temporal drift, 480k users |
| **Spotify Million Playlist** | Music playlist continuation | Cold-start on new playlists, sequential order matters |

**Loading Amazon Reviews (drop-in replacement)**

```python
import json, gzip

def load_amazon(path):
    with gzip.open(path) as f:
        for line in f:
            yield json.loads(line)

# Filter: rating >= 4 → positive, same as MovieLens pattern
rows = [(d['reviewerID'], d['asin'], d['overall'])
        for d in load_amazon('reviews_Electronics_5.json.gz')
        if d['overall'] >= 4]
df = pd.DataFrame(rows, columns=['user_id', 'item_id', 'rating'])
```

**What production conditions each dataset stresses**

```
CONDITION                    DATASET TO USE
─────────────────────────────────────────────────────────
Scale (millions of items)    Amazon, Netflix Prize, Pinterest
Implicit-only signal         Steam, LastFM, Pinterest, Criteo
Cold-start (new items)       Amazon (long tail of rare products)
Temporal drift               Netflix Prize, MovieLens 20M (use time split)
Sequential behaviour         Spotify, LastFM (order of interactions matters)
Side features (text/image)   Amazon (review text), Pinterest (images)
Extreme sparsity             Criteo, Pinterest
```

**Recommended progression**

1. Start: MovieLens 1M — same code, 10× data
2. Add noise: Amazon Electronics — implicit signals, sparse
3. Test scale: MovieLens 20M with temporal train/val/test split
4. Stress ANN: Pinterest or Amazon at full scale — build a real Faiss index

---

### Q20: How does final ranking work in production? Do systems use LightGBM or XGBoost?

Yes — LightGBM and XGBoost are standard for the re-ranking stage. Production recommendation systems are always multi-stage pipelines. The Two-Tower + ANN is just the first stage (retrieval). Here is the full picture:

```
MULTI-STAGE PRODUCTION PIPELINE
─────────────────────────────────────────────────────────────────────

STAGE 1 — CANDIDATE RETRIEVAL  (fast, approximate, high recall)
────────────────────────────────
Multiple candidate generators run in parallel, each targeting a
different aspect of user preference:

  Two-Tower ANN        →  top-200  (collaborative: "users like you liked")
  Content-based ANN    →  top-200  (item features: "similar to what you watched")
  Trending / Popular   →  top-100  (global or category popularity)
  Session-based model  →  top-100  (recent browsing context)
  Editorial / Rules    →  top-50   (new releases, promoted items)
        │
        ▼
  Merge + deduplicate  →  ~500–1000 unique candidates
        │
        ▼
  Each candidate is a (user, item) pair with no final score yet


STAGE 2 — SCORING / RE-RANKING  (slower, precise, feature-rich)
─────────────────────────────────
For each of the ~500–1000 (user, item) candidates, build a feature
vector and run a scoring model:

  Features per candidate:
    ├── user features     : demographics, tenure, avg rating, device
    ├── item features     : genre, release year, avg CTR, freshness
    ├── interaction       : dot product score from stage 1, which
    │                       generator produced this candidate
    ├── cross features    : (user_genre_affinity × item_genre),
    │                       (user_activity_level × item_popularity)
    └── contextual        : time of day, day of week, platform

  Scoring model options:
    ├── LightGBM / XGBoost   ← most common in industry
    │     fast inference, handles heterogeneous features well,
    │     interpretable, robust to missing values
    │
    ├── Deep & Cross Network (DCN)
    │     neural model, learns explicit feature crosses
    │
    └── DIN / DIEN (Alibaba)
          attention over user's historical interactions,
          weights history by similarity to current candidate

  Output: one score per candidate  (predicted CTR, engagement, rating)


STAGE 3 — FINAL RANKING + BUSINESS RULES
──────────────────────────────────────────
Sort candidates by score from stage 2, then apply constraints:

  Diversity       : don't show 10 action films in a row
                    (MMR — Maximal Marginal Relevance)
  Freshness       : boost items uploaded in last 7 days
  Deduplication   : remove items user already watched / purchased
  Sponsored slots : inject paid placements at fixed positions
  Fairness        : ensure minority-category items get minimum exposure
  Business rules  : don't show age-restricted content to minors
        │
        ▼
  Final ranked list of top-10 (or top-N) items served to user
```

**Why not use the Two-Tower score directly for final ranking?**

The Two-Tower score is a single dot product — it only captures collaborative similarity (users-like-you). It has no access to item freshness, current CTR, contextual signals, or business constraints. LightGBM/XGBoost can incorporate all of these as features and is trained directly on the target metric (CTR, watch time, purchase) rather than triplet loss.

```
TWO-TOWER SCORE              RE-RANKER SCORE
────────────────             ───────────────────────────────────
1 number: cos(θ)             1 number: predicted CTR / engagement
based on: ID embeddings      based on: embeddings + metadata +
          only                         context + business signals
trained on: triplet loss     trained on: actual user click labels
speed: microseconds          speed: milliseconds (500 candidates)
recall-oriented              precision-oriented
```

**Why LightGBM / XGBoost specifically?**

- Heterogeneous features: mixes dense embeddings, sparse categoricals, and floats without normalisation
- Fast inference: scores 500 candidates in ~1ms on CPU
- Calibrated probabilities: outputs actual CTR estimates, not just ranks
- Interpretable: SHAP values explain why an item was ranked highly
- Battle-tested: Netflix, Airbnb, LinkedIn, Twitter all use GBDT re-rankers

**The full stack in one line per stage**

```
Retrieval (Two-Tower ANN)  →  ~500 candidates  in ~10ms
Re-ranking (LightGBM)      →  scored list       in ~5ms
Business rules             →  final top-10      in ~1ms
Total serving latency      →  ~20ms end-to-end
```

---

### Q19: How would you take this to production?

1. Pre-compute all item embeddings and store them in a vector database (e.g. pgvector, Pinecone, Faiss).
2. At request time, run only the user tower → query the vector DB for nearest neighbours.
3. Retrain periodically (daily/weekly) as new interaction data arrives.
4. For new users with no history, fall back to popularity-based or content-based recommendations until enough interactions accumulate.

---
---

## FAQ — minerU_2.5.ipynb

### Q1: What is MinerU 2.5?

MinerU is an open-source document intelligence toolkit from OpenDataLab. Version 2.5 adds a vision-language (VL) pipeline powered by a fine-tuned Qwen2-VL model. It extracts structured content from document images — titles, headers, body text, tables, figures — and returns them as typed blocks with normalised bounding boxes. It works on scanned PDFs, photos of documents, receipts, ID cards, and technical reports.

---

### Q2: What model does it actually use under the hood?

`opendatalab/MinerU2.5-2509-1.2B` — a 1.2 billion parameter vision-language model fine-tuned from Qwen2-VL on document understanding tasks. "2509" in the name is the training checkpoint date (September 2025). It's loaded via HuggingFace Transformers:

```python
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "opendatalab/MinerU2.5-2509-1.2B", dtype="auto", device_map="auto"
)
```

`device_map="auto"` places the model on GPU if available, CPU otherwise. `dtype="auto"` lets HuggingFace pick the best precision (bfloat16 on Ampere+ GPUs).

---

### Q3: What does `client.two_step_extract()` do?

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

### Q4: What block types does MinerU detect and what color is each in the visualisation?

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

### Q5: What are "normalised bounding boxes" and how are they converted to pixels?

The model returns bounding boxes as `[x1, y1, x2, y2]` with values in `[0, 1]` relative to the image dimensions. This makes them resolution-independent. `display_extraction` converts to pixel coordinates for drawing:

```python
x1, y1, x2, y2 = bbox
rect_coords = [x1 * width, y1 * height, x2 * width, y2 * height]
draw.rectangle(rect_coords, outline=color, width=3)
```

A bbox of `[0.1, 0.05, 0.9, 0.12]` on a 1000×800 image becomes the rectangle `[100, 40, 900, 96]` in pixels.

---

### Q6: What documents were tested in the notebook?

Four document types from the `ocr-documents.zip` archive:

| File | Document type |
|---|---|
| `nvidia-first-page.jpg` | Corporate annual report cover page |
| `nvidia-inner-page.jpg` | Dense financial/technical inner page with tables |
| `receipt.jpg` | Retail receipt |
| `id-card.png` | ID card |

This range tests MinerU across very different layouts — structured corporate docs, semi-structured receipts, and identity documents.

---

### Q7: What is `MinerUClient` and why use it instead of calling the model directly?

`MinerUClient` from `mineru-vl-utils` is a thin wrapper that handles the two-step pipeline (layout detection → content extraction) and manages prompt construction for the Qwen2-VL model. Calling the model directly would require you to manually craft the VL prompt, parse the raw output, and convert coordinates — `MinerUClient` abstracts all of that into a single `two_step_extract(image)` call.

---

### Q8: What is `magic-pdf.json` and why is it configured?

MinerU's CLI tool (`mineru`) reads `~/.config/magic-pdf.json` for runtime settings. The notebook sets `device-mode: "cuda"` and points `models-dir` at the HuggingFace cache path so the CLI knows where the downloaded model lives and runs it on GPU:

```python
config_data = {
    "device-mode": "cuda",
    "models-dir": "C:\\Users\\rohan\\.cache\\huggingface\\hub\\models--opendatalab--MinerU2.5-2509-1.2B"
}
```

This only matters when using the CLI (`mineru -p input.pdf`). The Python API (`MinerUClient`) doesn't read this file.

---

### Q9: How do I run MinerU on a PDF from the command line?

```bash
# GPU
CUDA_VISIBLE_DEVICES=0 mineru -p /path/to/input.pdf -o /path/to/output/

# CPU only
mineru -p /path/to/input.pdf -o /path/to/output/ --device cpu
```

The output directory will contain Markdown, JSON (structured blocks), and images for each extracted figure/table. The `CUDA_VISIBLE_DEVICES=0` flag selects the first GPU.

---

### Q10: How does the side-by-side display work?

`display_extraction` produces an inline HTML page rendered directly in the Jupyter output cell:

- **Left pane** — the original image with coloured bounding boxes and type labels drawn on it using `PIL.ImageDraw`, then base64-encoded and embedded as a `data:image/png` URI (no file I/O needed).
- **Right pane** — the extracted text rendered as HTML: titles become `<h2>`, headers `<h4>`, tables are injected as raw `<table>` HTML, everything else is `<p>`. The pane has `overflow-y: auto` so long documents scroll.

---

### Q11: Does MinerU work without a GPU?

Yes, but slowly. The 1.2B parameter model runs on CPU; expect seconds to minutes per page depending on hardware. The notebook checks GPU availability and `magic-pdf.json` sets `device-mode: "cuda"`. For CPU-only use set `device-mode: "cpu"` in the config and remove `CUDA_VISIBLE_DEVICES` from the CLI call. The Python API respects `device_map="auto"` which falls back to CPU automatically.

---

### Q12: Why does the notebook call `!pip install -U "mineru[all]"` twice (cell 2 and cell 23)?

Cell 23 is a leftover re-run of the install cell, likely from debugging a missing dependency mid-session. It's harmless (pip is idempotent) but redundant — it can be deleted.

---

### Q13: How do I use MinerU on my own documents?

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
