"""LMBN fine-tune script for Colab.

Paste the contents of this file into a single Colab cell after running:

    !tar xzf lmbn_ft_bundle.tgz
    !pip install -q torch torchvision pillow tqdm scikit-learn

Then run the cell. Training takes ~30-60 min on a T4 GPU.

Outputs:
- ``lmbn_n_finetuned.pth`` — fine-tuned state_dict, ready for download
- Prints intra/inter cosine sim on held-out IDs every 5 epochs
- Saves best checkpoint (highest intra - inter delta) automatically
"""
from __future__ import annotations

import os
import sys
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

# ---- Add bundled model code to import path ------------------------------------
sys.path.insert(0, str(Path("models").resolve().parent))
from models.lightmbn_n import LMBN_n  # noqa: E402

# ---- Reproducibility ---------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ---- Config ------------------------------------------------------------------
DATA_ROOT = Path("dataset")
# Public ReID diversity base. Point at Market-1501's bounding_box_train (flat
# files named <pid>_cN sNN_*.jpg). Set MARKET_ROOT="" to train target-only.
# Only 12 target IDs would overfit; Market-1501's 751 IDs teach general,
# scale-invariant person discrimination that the target IDs then adapt.
MARKET_ROOT = Path(os.environ.get("MARKET_ROOT", "market1501/bounding_box_train"))
MARKET_PRIORITY_PER_BATCH = 4  # of BATCH_P IDs/batch, force this many to be target IDs (oversample the small target set)
INIT_WEIGHTS = "lmbn_n_cuhk03_d.pth"
OUTPUT_WEIGHTS = "lmbn_n_finetuned.pth"
HELD_OUT_IDS = {"person11", "person12"}  # legacy unseen-ID split; superseded by d02/d03 cross-quality
INPUT_H, INPUT_W = 256, 128  # must match inference_engine resize (model_registry.py)
VAL_QUALITY_TAGS = {"d02", "d03"}  # filename tag dDD; held out as harder val set
BATCH_P = 8   # IDs per batch
BATCH_K = 4   # crops per ID per batch
NUM_EPOCHS = 50
LR_HEADS = 3e-4
WEIGHT_DECAY = 5e-4
LR_DROPS = [30, 40]
TRIPLET_MARGIN = 0.3
LOSS_WEIGHT_TRI = 1.0
LOSS_WEIGHT_CE = 1.0
EVAL_EVERY = 5

# ---- Dataset -----------------------------------------------------------------
def _quality_tag(path: Path) -> str:
    """Extract dDD quality bucket from edge-service filename: vidN_fNNNNNN_dDD_cCC.CC.jpg."""
    for token in path.stem.split("_"):
        if len(token) >= 2 and token[0] == "d" and token[1:].isdigit():
            return token
    return "d00"


# Market-1501 ids to drop: -1 = junk/background detections, 0000 = distractors.
_MARKET_SKIP_PIDS = {"-1", "0000"}


def market_pids(market_root: Path) -> list[str]:
    """Distinct Market-1501 person-ids present in a flat image folder."""
    pids = set()
    for p in market_root.iterdir():
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        pid = p.name.split("_", 1)[0]
        if pid and pid not in _MARKET_SKIP_PIDS:
            pids.add(pid)
    return sorted(pids)


def load_market_samples(market_root: Path, id_to_idx: dict[str, int], prefix: str = "mkt_") -> list[tuple[Path, int]]:
    """(path, label_idx) for every valid Market-1501 crop, mapped to the shared
    label space via ``prefix+pid`` keys already present in ``id_to_idx``."""
    samples: list[tuple[Path, int]] = []
    for p in sorted(market_root.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        pid = p.name.split("_", 1)[0]
        if pid in _MARKET_SKIP_PIDS:
            continue
        key = prefix + pid
        idx = id_to_idx.get(key)
        if idx is not None:
            samples.append((p, idx))
    return samples


class RandomResolutionDegradation:
    """Simulate distant / low-detail crops by downscaling then upscaling back.

    The cross-scale failure we're fixing: same person looks very different to the
    embedding when far (tiny, low-detail) vs near (large, sharp). Most labeled
    crops are near/clear (d00/d01), so we synthesize the *far* look on the fly —
    shrink the crop to a random fraction then blow it back up to INPUT size,
    destroying high-frequency detail the way a distant detection would. Forces the
    triplet loss to pull near↔synthetic-far of the same ID together → scale
    invariance. Applied on the already-resized (INPUT_H, INPUT_W) PIL image.
    """

    def __init__(self, p: float = 0.5, min_scale: float = 0.2, max_scale: float = 0.6):
        self.p = p
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        w, h = img.size
        f = random.uniform(self.min_scale, self.max_scale)
        sw, sh = max(1, int(round(w * f))), max(1, int(round(h * f)))
        small = img.resize((sw, sh), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


class ReIDDataset(Dataset):
    def __init__(self, root: Path, train: bool, id_filter: set[str] | None = None,
                 quality_filter: set[str] | None = None, quality_exclude: set[str] | None = None,
                 id_to_idx: dict[str, int] | None = None,
                 extra_samples: list[tuple[Path, int]] | None = None):
        self.samples: list[tuple[Path, int]] = []
        ids = sorted([d.name for d in root.iterdir() if d.is_dir()])
        if id_filter is not None:
            ids = [i for i in ids if i in id_filter]
        # Shared id_to_idx so train + val (cross-quality on the same IDs) align.
        self.id_to_idx = id_to_idx if id_to_idx is not None else {name: idx for idx, name in enumerate(ids)}
        skipped_bad = 0
        for name in ids:
            for img_path in sorted((root / name).iterdir()):
                if img_path.name.startswith(".") or img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                tag = _quality_tag(img_path)
                if quality_filter is not None and tag not in quality_filter:
                    continue
                if quality_exclude is not None and tag in quality_exclude:
                    continue
                # Probe-decode once so dataloader workers don't crash on bad files.
                try:
                    with Image.open(img_path) as im:
                        im.verify()
                except Exception:
                    skipped_bad += 1
                    continue
                self.samples.append((img_path, self.id_to_idx[name]))
        # Append samples from another source (e.g. Market-1501) already mapped to
        # the shared label space — lets train mix public-diversity IDs with target.
        if extra_samples:
            self.samples.extend(extra_samples)
        if skipped_bad:
            print(f"[{'train' if train else 'val'}] skipped {skipped_bad} undecodable files")
        self.train = train
        if train:
            # Production-matched augmentation: blur + scale jitter + erasing simulate
            # the motion / JPEG / partial-occlusion conditions the live pipeline feeds.
            self.tf = transforms.Compose([
                transforms.Resize((INPUT_H, INPUT_W)),
                # Cross-scale invariance: turn near/clear crops into synthetic far/
                # low-detail ones so same-ID near↔far get pulled together.
                RandomResolutionDegradation(p=0.5, min_scale=0.2, max_scale=0.6),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))], p=0.3
                ),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.7, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((INPUT_H, INPUT_W)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        print(f"[{'train' if train else 'val'}] IDs={len(ids)} samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # Recover by returning a neighbor sample — keeps the batch shape stable.
            return self.__getitem__((idx + 1) % len(self.samples))
        return self.tf(img), label


class PKSampler(Sampler):
    """PK sampler: each batch has P distinct IDs × K crops/ID.

    Crops are sampled with replacement for IDs with fewer than K crops, so
    every ID can contribute regardless of folder size.
    """

    def __init__(self, dataset: ReIDDataset, p: int, k: int, num_batches: int,
                 priority_ids: set[int] | None = None, priority_per_batch: int = 0):
        self.dataset = dataset
        self.p = p
        self.k = k
        self.num_batches = num_batches
        self.id_to_indices: dict[int, list[int]] = defaultdict(list)
        for i, (_, label) in enumerate(dataset.samples):
            self.id_to_indices[label].append(i)
        self.ids = list(self.id_to_indices.keys())
        if len(self.ids) < p:
            raise ValueError(f"Need ≥ {p} IDs, have {len(self.ids)}")
        # Oversample a small priority subset (target IDs) so they appear every
        # batch despite being vastly outnumbered by Market-1501 IDs — otherwise
        # random P-of-763 sampling almost never touches the 12 target IDs.
        self.priority_ids = [i for i in (priority_ids or set()) if i in self.id_to_indices]
        self.other_ids = [i for i in self.ids if i not in set(self.priority_ids)]
        self.priority_per_batch = min(priority_per_batch, len(self.priority_ids), p)

    def __iter__(self):
        for _ in range(self.num_batches):
            if self.priority_per_batch > 0 and len(self.other_ids) >= self.p - self.priority_per_batch:
                chosen_ids = (random.sample(self.priority_ids, self.priority_per_batch)
                              + random.sample(self.other_ids, self.p - self.priority_per_batch))
            else:
                chosen_ids = random.sample(self.ids, self.p)
            batch: list[int] = []
            for pid in chosen_ids:
                pool = self.id_to_indices[pid]
                if len(pool) >= self.k:
                    batch.extend(random.sample(pool, self.k))
                else:
                    batch.extend(random.choices(pool, k=self.k))
            yield batch

    def __len__(self):
        return self.num_batches


# ---- Model build & init ------------------------------------------------------
# Train = target d00/d01 (good crops) ⊕ all Market-1501 (public diversity, anti-
# overfit). Val = target d02/d03 (far/messy — the cross-scale case we care about).
# Same target IDs on both sides; Market only on train. Unified label space.
target_ids = sorted([d.name for d in DATA_ROOT.iterdir() if d.is_dir()])

mkt_keys: list[str] = []
if str(MARKET_ROOT) and MARKET_ROOT.exists():
    mkt_keys = ["mkt_" + pid for pid in market_pids(MARKET_ROOT)]
    print(f"Market-1501: {len(mkt_keys)} IDs from {MARKET_ROOT}")
else:
    print(f"Market-1501 not found at '{MARKET_ROOT}' — training TARGET-ONLY ({len(target_ids)} IDs); overfit risk.")

id_to_idx = {name: idx for idx, name in enumerate(target_ids + mkt_keys)}
target_idx_set = {id_to_idx[n] for n in target_ids}
market_samples = load_market_samples(MARKET_ROOT, id_to_idx) if mkt_keys else []

train_ds = ReIDDataset(
    DATA_ROOT, train=True, id_filter=set(target_ids),
    quality_exclude=VAL_QUALITY_TAGS, id_to_idx=id_to_idx,
    extra_samples=market_samples,
)
val_ds = ReIDDataset(
    DATA_ROOT, train=False, id_filter=set(target_ids),
    quality_filter=VAL_QUALITY_TAGS, id_to_idx=id_to_idx,
)

NUM_TRAIN_IDS = len(id_to_idx)
print(f"Training on {NUM_TRAIN_IDS} IDs ({len(target_ids)} target + {len(mkt_keys)} market) "
      f"| train crops={len(train_ds)} (market {len(market_samples)}) | val crops={len(val_ds)}")
if len(val_ds) < 10:
    print(f"WARNING: only {len(val_ds)} val crops — d02/d03 tags rare. Falling back to held-out target IDs.")
    train_ids = {n for n in target_ids if n not in HELD_OUT_IDS}
    train_id_to_idx = {n: i for i, n in enumerate(sorted(train_ids))}
    val_id_to_idx = {n: i for i, n in enumerate(sorted(HELD_OUT_IDS))}
    train_ds = ReIDDataset(DATA_ROOT, train=True, id_filter=train_ids, id_to_idx=train_id_to_idx)
    val_ds = ReIDDataset(DATA_ROOT, train=False, id_filter=HELD_OUT_IDS, id_to_idx=val_id_to_idx)
    NUM_TRAIN_IDS = len(train_id_to_idx)
    target_idx_set = set(train_id_to_idx.values())

# Build model. LMBN_n's classifier heads have shape (num_classes, 512). The
# CUHK03-D checkpoint was trained on 767 IDs — its classifier weights won't
# transfer, but the embedding/BNNeck weights will. We initialize at 767, load
# weights strictly, then replace the classifier heads with NUM_TRAIN_IDS-class
# heads for fine-tuning. The embedding output (the part actually used at
# inference) is NOT affected by the classifier shape.
model = LMBN_n(num_classes=767, feats=512, activation_map=False)

ckpt = torch.load(INIT_WEIGHTS, map_location="cpu", weights_only=False)
state_dict = ckpt
if isinstance(ckpt, dict):
    for key in ("state_dict", "net", "model"):
        if key in ckpt and isinstance(ckpt[key], dict):
            state_dict = ckpt[key]
            break

def _strip(k):
    for p in ("module.", "model."):
        if k.startswith(p):
            k = k[len(p):]
    return k

state_dict = {_strip(k): v for k, v in state_dict.items()}
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Loaded init weights — missing={len(missing)} unexpected={len(unexpected)}")

# Swap classifier heads to NUM_TRAIN_IDS classes (10 train IDs).
def _swap_classifier(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and child.out_features == 767:
            setattr(module, name, nn.Linear(child.in_features, NUM_TRAIN_IDS, bias=False))
        else:
            _swap_classifier(child)

_swap_classifier(model)

# Freeze backbone (OSNet branches). Train only the heads + BNNeck + attention.
FREEZE_PREFIXES = ("backone.", "global_branch.", "partial_branch.", "channel_branch.")
trainable, frozen = 0, 0
for name, param in model.named_parameters():
    if any(name.startswith(p) for p in FREEZE_PREFIXES):
        param.requires_grad = False
        frozen += param.numel()
    else:
        trainable += param.numel()
print(f"Trainable params: {trainable:,}  Frozen: {frozen:,}")

model = model.to(DEVICE)

# ---- Loss --------------------------------------------------------------------
class BatchHardTripletLoss(nn.Module):
    """Standard batch-hard triplet on L2-normalized features."""

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feats = F.normalize(feats, dim=1)
        sim = feats @ feats.t()
        dist = 1.0 - sim  # cosine distance, in [0, 2]
        same = labels.unsqueeze(0) == labels.unsqueeze(1)
        diff = ~same
        # Mask self with -inf for hardest-positive (max dist in same-class)
        dist_pos = dist.masked_fill(~same, -math.inf)
        dist_pos.fill_diagonal_(-math.inf)
        hardest_pos, _ = dist_pos.max(dim=1)
        dist_neg = dist.masked_fill(~diff, math.inf)
        hardest_neg, _ = dist_neg.min(dim=1)
        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        # Ignore rows where there was no positive (singleton class in batch)
        valid = torch.isfinite(hardest_pos)
        if valid.sum() == 0:
            return torch.zeros((), device=feats.device, requires_grad=True)
        return loss[valid].mean()


triplet_loss = BatchHardTripletLoss(margin=TRIPLET_MARGIN)
ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

# ---- Optimizer ---------------------------------------------------------------
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=LR_HEADS, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=LR_DROPS, gamma=0.1)


# ---- Forward helper ----------------------------------------------------------
def model_forward(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Returns (embedding, logits_list).

    LMBN_n's forward returns either embeddings (eval) or (embeddings, logits)
    (train, when classifier heads attached). We always reach inside to grab the
    pre-pooled feature stack so we can compute triplet on the concatenated
    embedding regardless of mode.
    """
    out = model(x)
    # Eval mode: torch.stack of 7 BN features along dim=2 → (B, 512, 7). Collapse.
    if isinstance(out, torch.Tensor):
        feats = out.mean(dim=2) if out.dim() == 3 else out
        return feats, []
    # Train mode: (feats_list, logits_list). feats_list is 7 tensors of (B, 512);
    # mirror eval by stacking + averaging so triplet loss optimizes the same vector
    # the pipeline consumes at inference. logits_list is 3 logits used for CE.
    if isinstance(out, (list, tuple)) and len(out) >= 2:
        feats_list, logits_list = out[0], out[1]
        if isinstance(feats_list, list):
            emb = torch.stack(feats_list, dim=2).mean(dim=2)
        else:
            emb = feats_list.mean(dim=2) if feats_list.dim() == 3 else feats_list
        logits = [t for t in (logits_list or []) if t.dim() == 2 and t.size(1) == NUM_TRAIN_IDS]
        return emb, logits
    raise RuntimeError(f"Unknown LMBN_n output type: {type(out)}")


# ---- Eval --------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, val_ds) -> tuple[float, float]:
    model.eval()
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
    feats_all = []
    labels_all = []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        feats, _ = model_forward(model, imgs)
        feats = F.normalize(feats, dim=1).cpu()
        feats_all.append(feats)
        labels_all.append(labels)
    feats_all = torch.cat(feats_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    sim = feats_all @ feats_all.t()
    same = labels_all.unsqueeze(0) == labels_all.unsqueeze(1)
    diff = ~same
    iu = torch.triu_indices(len(feats_all), len(feats_all), offset=1)
    same_iu = same[iu[0], iu[1]]
    diff_iu = diff[iu[0], iu[1]]
    sim_iu = sim[iu[0], iu[1]]
    intra = sim_iu[same_iu].mean().item() if same_iu.any() else float("nan")
    inter = sim_iu[diff_iu].mean().item() if diff_iu.any() else float("nan")
    return intra, inter


# ---- Train loop --------------------------------------------------------------
sampler = PKSampler(
    train_ds, p=BATCH_P, k=BATCH_K,
    num_batches=max(20, len(train_ds) // (BATCH_P * BATCH_K)),
    priority_ids=target_idx_set,
    priority_per_batch=(MARKET_PRIORITY_PER_BATCH if mkt_keys else 0),
)
loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=2)

best_delta = -math.inf
best_state = None
print("\n=== Pre-training eval (CUHK03-D baseline) ===")
intra0, inter0 = evaluate(model, val_ds)
print(f"intra={intra0:.4f}  inter={inter0:.4f}  delta={intra0 - inter0:+.4f}")

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    losses = []
    t0 = time.time()
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        feats, logits = model_forward(model, imgs)
        loss_tri = triplet_loss(feats, labels)
        loss_ce = sum(ce_loss(l, labels) for l in logits) / max(1, len(logits)) if logits else feats.new_zeros(())
        loss = LOSS_WEIGHT_TRI * loss_tri + LOSS_WEIGHT_CE * loss_ce
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    scheduler.step()
    avg = sum(losses) / max(1, len(losses))
    print(f"epoch {epoch:02d} | loss {avg:.4f} | lr {optimizer.param_groups[0]['lr']:.2e} | {time.time()-t0:.1f}s", end="")
    if epoch % EVAL_EVERY == 0 or epoch == NUM_EPOCHS:
        intra, inter = evaluate(model, val_ds)
        delta = intra - inter
        print(f"  | val intra={intra:.4f} inter={inter:.4f} delta={delta:+.4f}", end="")
        if delta > best_delta:
            best_delta = delta
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print("  ★ new best")
        else:
            print()
    else:
        print()

# ---- Save --------------------------------------------------------------------
if best_state is None:
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
torch.save(best_state, OUTPUT_WEIGHTS)
print(f"\nSaved {OUTPUT_WEIGHTS}  (best held-out delta = {best_delta:+.4f})")
print(f"Pre-training baseline delta was {intra0 - inter0:+.4f}")
print(f"Improvement: {(best_delta - (intra0 - inter0)):+.4f}")
print(f"\nNumber of train IDs (use this for num_classes in model_registry.py): {NUM_TRAIN_IDS}")
