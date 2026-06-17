#!/usr/bin/env python3
# depth_plane_sim.py

import argparse, csv, json, math, os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Plane:
    a: float
    b: float
    c: float
    cx: float
    cy: float


@dataclass
class ModeResult:
    mode: str
    candidate_name: str
    plane: Plane
    recon_block: np.ndarray
    bits: float
    sse: float
    cost: float
    q_values: Tuple[int, ...]


@dataclass
class LeafRecord:
    x: int
    y: int
    w: int
    h: int
    plane: Plane


@dataclass
class CSNode:
    x: int
    y: int
    w: int
    h: int
    depth: int
    parent: Optional["CSNode"] = None
    split: str = "leaf"  # leaf, qt, bh, bv
    children: List["CSNode"] = field(default_factory=list)

    best: Optional[ModeResult] = None
    actual: Optional[Plane] = None
    avail_modes: List[str] = field(default_factory=list)
    avail_cands: List[str] = field(default_factory=list)

    bits: float = 0.0
    sse: float = 0.0
    cost: float = 0.0
    split_bits: float = 0.0
    qt_flag_present: bool = False

    def is_leaf(self):
        return self.split == "leaf"


class AdaptiveProbTable:
    def __init__(
        self,
        symbols,
        init_probs=None,
        update_rate=0.05,
        p_min=0.02,
        p_max=0.95,
        name="",
    ):
        self.symbols = list(symbols)
        self.n = len(self.symbols)
        self.update_rate = float(update_rate)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.name = name

        if self.n <= 0:
            raise ValueError("AdaptiveProbTable needs symbols")
        if self.p_min * self.n > 1.0:
            raise ValueError(f"{name}: p_min too large")
        if self.p_max * self.n < 1.0:
            raise ValueError(f"{name}: p_max too small")

        if init_probs is None:
            self.probs = {s: 1.0 / self.n for s in self.symbols}
        else:
            total = sum(float(init_probs.get(s, 0.0)) for s in self.symbols)
            if total <= 0:
                raise ValueError("init_probs sum must be positive")
            self.probs = {
                s: float(init_probs.get(s, 0.0)) / total for s in self.symbols
            }

        self._project()

    def bits(self, symbol, available_symbols=None):
        if symbol not in self.probs:
            raise KeyError(f"unknown symbol {symbol} in {self.name}")

        if available_symbols is None:
            return -math.log2(max(self.probs[symbol], 1e-12))

        av = [s for s in available_symbols if s in self.probs]

        if symbol not in av:
            raise KeyError(f"{symbol} not available in {self.name}")

        if len(av) <= 1:
            return 0.0

        norm = sum(self.probs[s] for s in av)
        p = self.probs[symbol] / norm if norm > 0 else 1.0 / len(av)
        return -math.log2(max(p, 1e-12))

    def update(self, selected):
        if selected not in self.probs:
            raise KeyError(f"unknown selected {selected} in {self.name}")

        psel = min(self.p_max, 1.0 - (self.n - 1) * self.p_min)
        others = [s for s in self.symbols if s != selected]
        target = {selected: psel}

        for s in others:
            target[s] = (1.0 - psel) / len(others) if others else 0.0

        lr = self.update_rate

        for s in self.symbols:
            self.probs[s] = (1.0 - lr) * self.probs[s] + lr * target[s]

        self._project()

    def _project(self):
        for s in self.symbols:
            self.probs[s] = min(max(self.probs[s], self.p_min), self.p_max)

        for _ in range(64):
            diff = 1.0 - sum(self.probs.values())

            if abs(diff) < 1e-12:
                break

            if diff > 0:
                adj = [
                    s for s in self.symbols
                    if self.probs[s] < self.p_max - 1e-12
                ]
            else:
                adj = [
                    s for s in self.symbols
                    if self.probs[s] > self.p_min + 1e-12
                ]

            if not adj:
                break

            add = diff / len(adj)

            for s in adj:
                self.probs[s] = min(max(self.probs[s] + add, self.p_min), self.p_max)

    def snapshot(self, prefix):
        return {f"{prefix}_{s}_prob": self.probs[s] for s in self.symbols}


class BinaryAdaptiveProb:
    def __init__(
        self,
        init_p1=0.5,
        update_rate=0.05,
        p_min=0.02,
        p_max=0.98,
        name="",
    ):
        self.p1 = float(init_p1)
        self.update_rate = float(update_rate)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.name = name
        self._clip()

    def _clip(self):
        self.p1 = min(max(self.p1, self.p_min), self.p_max)

    def bits(self, b):
        if b not in (0, 1):
            raise ValueError("bin must be 0/1")

        p = self.p1 if b else 1.0 - self.p1
        return -math.log2(max(p, 1e-12))

    def update(self, b):
        if b not in (0, 1):
            raise ValueError("bin must be 0/1")

        target = self.p_max if b else self.p_min
        self.p1 = (1.0 - self.update_rate) * self.p1 + self.update_rate * target
        self._clip()


def unary_candidate_bits(idx, n, ctx, truncated=True):
    if n <= 1:
        return 0.0

    if idx < 0 or idx >= n or n > len(ctx):
        raise ValueError("bad unary candidate")

    bits = sum(ctx[i].bits(0) for i in range(idx))

    if not (truncated and idx == n - 1):
        bits += ctx[idx].bits(1)

    return bits


def unary_candidate_update(idx, n, ctx, truncated=True):
    if n <= 1:
        return

    if idx < 0 or idx >= n or n > len(ctx):
        raise ValueError("bad unary candidate")

    for i in range(idx):
        ctx[i].update(0)

    if not (truncated and idx == n - 1):
        ctx[idx].update(1)


def qt_split_flag_bits(adaptive, depth, flag):
    if adaptive is not None and "qt_split" in adaptive and depth < len(adaptive["qt_split"]):
        return adaptive["qt_split"][depth].bits(flag)

    return 1.0


def qt_split_flag_update(adaptive, node):
    if adaptive is None or "qt_split" not in adaptive:
        return

    if not node.qt_flag_present:
        return

    if node.depth >= len(adaptive["qt_split"]):
        return

    flag = 1 if node.split == "qt" else 0
    adaptive["qt_split"][node.depth].update(flag)


def ceil_log2(x):
    return 0 if x <= 1 else int(math.ceil(math.log2(x)))


def exp_golomb_len_unsigned(u):
    if u < 0:
        raise ValueError("ue input negative")

    return 2 * int(math.floor(math.log2(u + 1))) + 1


def signed_to_code_num(v):
    if v == 0:
        return 0

    return 2 * v - 1 if v > 0 else -2 * v


def exp_golomb_len_signed(v):
    return exp_golomb_len_unsigned(signed_to_code_num(v))


def quantize(x, q):
    return int(np.rint(x / q))


def dequantize(v, q):
    return float(v) * q


def adaptive_signed_residual_bits(q, model, abs_max):
    a = abs(q)

    if a <= abs_max:
        bits = model.bits(a)
    else:
        bits = model.bits("esc")
        bits += exp_golomb_len_unsigned(a - (abs_max + 1))

    if a > 0:
        bits += 1.0

    return bits


def adaptive_signed_residual_update(q, model, abs_max):
    model.update(abs(q) if abs(q) <= abs_max else "esc")


def create_adaptive_models(args):
    cand_symbols = [
        "temporal",
        "left",
        "top",
        "top_left",
        "top_right",
        "avg_left_top",
    ]

    models = {
        "mode": AdaptiveProbTable(
            ["direct", "copy", "delta"],
            update_rate=args.prob_lr,
            p_min=args.prob_min,
            p_max=args.prob_max,
            name="mode",
        ),
        "candidate": AdaptiveProbTable(
            cand_symbols,
            update_rate=args.prob_lr,
            p_min=args.prob_min,
            p_max=args.prob_max,
            name="candidate",
        ),
        "delta_abs_max": args.delta_abs_max,
    }

    if args.copy_candidate_unary:
        models["copy_candidate_unary"] = [
            BinaryAdaptiveProb(
                update_rate=args.prob_lr,
                p_min=args.prob_min,
                p_max=args.prob_max,
                name=f"copy_ctx{i}",
            )
            for i in range(args.max_candidates)
        ]

    if args.qt_split_adaptive:
        models["qt_split"] = [
            BinaryAdaptiveProb(
                init_p1=0.5,
                update_rate=args.prob_lr,
                p_min=args.prob_min,
                p_max=args.prob_max,
                name=f"qt_split_depth{i}",
            )
            for i in range(args.max_qt_depth)
        ]

    if args.delta_residual_adaptive:
        syms = list(range(args.delta_abs_max + 1)) + ["esc"]

        for k in "abc":
            models[f"delta_res_abs_{k}"] = AdaptiveProbTable(
                syms,
                update_rate=args.prob_lr,
                p_min=args.prob_min,
                p_max=args.prob_max,
                name=f"delta_res_abs_{k}",
            )

    return models


class GridCache:
    def __init__(self):
        self.cache = {}

    def get(self, w, h):
        key = (w, h)

        if key not in self.cache:
            xs = np.arange(w, dtype=np.float64) - (w - 1) / 2.0
            ys = np.arange(h, dtype=np.float64) - (h - 1) / 2.0
            xx, yy = np.meshgrid(xs, ys)

            A = np.stack(
                [xx.reshape(-1), yy.reshape(-1), np.ones(w * h)],
                axis=1,
            )

            self.cache[key] = (xx, yy, np.linalg.pinv(A))

        return self.cache[key]


def fit_plane(block, pinv, cx, cy):
    a, b, c = (pinv @ block.astype(np.float64).reshape(-1)).tolist()
    return Plane(a, b, c, cx, cy)


def plane_to_center(p, cx, cy):
    return Plane(
        p.a,
        p.b,
        p.c + p.a * (cx - p.cx) + p.b * (cy - p.cy),
        cx,
        cy,
    )


def render_plane(p, xx, yy, maxv):
    return np.clip(np.rint(p.a * xx + p.b * yy + p.c), 0, maxv).astype(np.float64)


def block_sse(orig, recon):
    d = orig.astype(np.float64) - recon.astype(np.float64)
    return float(np.sum(d * d))


def overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def best_left(store, x, y, w, h):
    best = None
    bo = 0

    for r in store:
        if r.x + r.w == x:
            o = overlap(r.y, r.y + r.h, y, y + h)

            if o > bo:
                best = r
                bo = o

    return best


def best_top(store, x, y, w, h):
    best = None
    bo = 0

    for r in store:
        if r.y + r.h == y:
            o = overlap(r.x, r.x + r.w, x, x + w)

            if o > bo:
                best = r
                bo = o

    return best


def top_left(store, x, y):
    for r in store:
        if r.x + r.w == x and r.y + r.h == y:
            return r

    return None


def top_right(store, x, y, w):
    for r in store:
        if r.x == x + w and r.y + r.h == y:
            return r

    return None


def temporal_center(prev_store, cx, cy):
    if not prev_store:
        return None

    for r in prev_store:
        if r.x <= cx < r.x + r.w and r.y <= cy < r.y + r.h:
            return r

    return None


def make_candidates(store, prev_store, x, y, w, h, cx, cy, max_cands, use_temporal):
    cand = []
    conv = {}

    if use_temporal:
        r = temporal_center(prev_store, cx, cy)

        if r is not None:
            p = plane_to_center(r.plane, cx, cy)
            conv["temporal"] = p
            cand.append(("temporal", p))

    items = [
        ("left", best_left(store, x, y, w, h)),
        ("top", best_top(store, x, y, w, h)),
        ("top_left", top_left(store, x, y)),
        ("top_right", top_right(store, x, y, w)),
    ]

    for name, r in items:
        if r is not None:
            p = plane_to_center(r.plane, cx, cy)
            conv[name] = p
            cand.append((name, p))

    if "left" in conv and "top" in conv:
        l = conv["left"]
        t = conv["top"]

        cand.append(
            (
                "avg_left_top",
                Plane(
                    0.5 * (l.a + t.a),
                    0.5 * (l.b + t.b),
                    0.5 * (l.c + t.c),
                    cx,
                    cy,
                ),
            )
        )

    return cand[:max_cands]


def eval_direct(block, actual, xx, yy, args, adaptive, avail_modes):
    qa = quantize(actual.a, args.qa)
    qb = quantize(actual.b, args.qb)
    qc = quantize(actual.c, args.qc)

    p = Plane(
        dequantize(qa, args.qa),
        dequantize(qb, args.qb),
        dequantize(qc, args.qc),
        actual.cx,
        actual.cy,
    )

    recon = render_plane(p, xx, yy, args.max_value)
    sse = block_sse(block, recon)

    if adaptive is not None:
        bits = adaptive["mode"].bits("direct", avail_modes)
    else:
        bits = float(args.mode_bits)

    bits += exp_golomb_len_signed(qa)
    bits += exp_golomb_len_signed(qb)

    if qc >= 0:
        bits += exp_golomb_len_unsigned(qc)
    else:
        bits += exp_golomb_len_signed(qc)

    return ModeResult(
        "direct",
        "none",
        p,
        recon,
        bits,
        sse,
        sse + args.lambda_rd * bits,
        (qa, qb, qc),
    )


def eval_copy(block, cands, xx, yy, args, adaptive, avail_modes, avail_cands):
    out = []

    for i, (name, p) in enumerate(cands):
        recon = render_plane(p, xx, yy, args.max_value)
        sse = block_sse(block, recon)

        if adaptive is None:
            bits = float(args.mode_bits + ceil_log2(len(cands)))
        else:
            bits = adaptive["mode"].bits("copy", avail_modes)

            if "copy_candidate_unary" in adaptive:
                bits += unary_candidate_bits(
                    i,
                    len(cands),
                    adaptive["copy_candidate_unary"],
                )
            else:
                bits += adaptive["candidate"].bits(name, avail_cands)

        out.append(
            ModeResult(
                "copy",
                name,
                p,
                recon,
                bits,
                sse,
                sse + args.lambda_rd * bits,
                (),
            )
        )

    return out


def eval_delta(block, actual, cands, xx, yy, args, adaptive, avail_modes, avail_cands):
    out = []

    for name, pred in cands:
        qda = quantize(actual.a - pred.a, args.qa)
        qdb = quantize(actual.b - pred.b, args.qb)
        qdc = quantize(actual.c - pred.c, args.qc)

        p = Plane(
            pred.a + dequantize(qda, args.qa),
            pred.b + dequantize(qdb, args.qb),
            pred.c + dequantize(qdc, args.qc),
            actual.cx,
            actual.cy,
        )

        recon = render_plane(p, xx, yy, args.max_value)
        sse = block_sse(block, recon)

        if adaptive is None:
            bits = float(args.mode_bits + ceil_log2(len(cands)))
        else:
            bits = adaptive["mode"].bits("delta", avail_modes)
            bits += adaptive["candidate"].bits(name, avail_cands)

        if adaptive is not None and "delta_res_abs_a" in adaptive:
            bits += adaptive_signed_residual_bits(
                qda,
                adaptive["delta_res_abs_a"],
                adaptive["delta_abs_max"],
            )
            bits += adaptive_signed_residual_bits(
                qdb,
                adaptive["delta_res_abs_b"],
                adaptive["delta_abs_max"],
            )
            bits += adaptive_signed_residual_bits(
                qdc,
                adaptive["delta_res_abs_c"],
                adaptive["delta_abs_max"],
            )
        else:
            bits += exp_golomb_len_signed(qda)
            bits += exp_golomb_len_signed(qdb)
            bits += exp_golomb_len_signed(qdc)

        out.append(
            ModeResult(
                "delta",
                name,
                p,
                recon,
                bits,
                sse,
                sse + args.lambda_rd * bits,
                (qda, qdb, qdc),
            )
        )

    return out


def eval_leaf(padded, x, y, w, h, depth, parent, args, grid, store, prev_store, adaptive):
    block = padded[y : y + h, x : x + w]

    cx = x + (w - 1) / 2.0
    cy = y + (h - 1) / 2.0

    xx, yy, pinv = grid.get(w, h)
    actual = fit_plane(block, pinv, cx, cy)

    cands = make_candidates(
        store=store,
        prev_store=prev_store,
        x=x,
        y=y,
        w=w,
        h=h,
        cx=cx,
        cy=cy,
        max_cands=args.max_candidates,
        use_temporal=args.temporal_candidate,
    )

    avail_cands = [n for n, _ in cands]
    avail_modes = ["direct", "copy", "delta"] if cands else ["direct"]

    modes = [eval_direct(block, actual, xx, yy, args, adaptive, avail_modes)]

    if cands:
        modes += eval_copy(block, cands, xx, yy, args, adaptive, avail_modes, avail_cands)
        modes += eval_delta(block, actual, cands, xx, yy, args, adaptive, avail_modes, avail_cands)

    best = min(modes, key=lambda r: r.cost)

    return CSNode(
        x=x,
        y=y,
        w=w,
        h=h,
        depth=depth,
        parent=parent,
        split="leaf",
        best=best,
        actual=actual,
        avail_modes=avail_modes,
        avail_cands=avail_cands,
        bits=best.bits,
        sse=best.sse,
        cost=best.cost,
    )


def add_leaves_to_store(node, store):
    if node.is_leaf():
        store.append(LeafRecord(node.x, node.y, node.w, node.h, node.best.plane))
        return

    for c in node.children:
        add_leaves_to_store(c, store)


def parent_node(x, y, w, h, depth, parent, split, split_bits, children, args, qt_flag_present):
    n = CSNode(
        x=x,
        y=y,
        w=w,
        h=h,
        depth=depth,
        parent=parent,
        split=split,
        children=children,
        split_bits=split_bits,
        qt_flag_present=qt_flag_present,
    )

    for c in children:
        c.parent = n

    n.bits = split_bits + sum(c.bits for c in children)
    n.sse = sum(c.sse for c in children)
    n.cost = args.lambda_rd * split_bits + sum(c.cost for c in children)

    return n


def encode_node(padded, x, y, w, h, depth, parent, args, grid, store, prev_store, adaptive):
    qt_ok = (
        depth < args.max_qt_depth
        and w >= 2
        and h >= 2
        and w % 2 == 0
        and h % 2 == 0
    )

    bh_ok = h >= 2 and h % 2 == 0
    bv_ok = w >= 2 and w % 2 == 0
    extra_ok = bh_ok or bv_ok

    cand = []

    # no split
    leaf = eval_leaf(
        padded,
        x,
        y,
        w,
        h,
        depth,
        parent,
        args,
        grid,
        store,
        prev_store,
        adaptive,
    )

    leaf.qt_flag_present = qt_ok
    leaf.split_bits = 0.0

    if qt_ok:
        leaf.split_bits += qt_split_flag_bits(adaptive, depth, 0)

    if extra_ok:
        leaf.split_bits += 1.0  # extra_split_flag = 0, fixed 50:50

    leaf.bits += leaf.split_bits
    leaf.cost += args.lambda_rd * leaf.split_bits
    cand.append(leaf)

    # binary horizontal: top/bottom
    if bh_ok:
        st = list(store)
        h0 = h // 2

        c0 = eval_leaf(
            padded,
            x,
            y,
            w,
            h0,
            depth + 1,
            None,
            args,
            grid,
            st,
            prev_store,
            adaptive,
        )
        add_leaves_to_store(c0, st)

        c1 = eval_leaf(
            padded,
            x,
            y + h0,
            w,
            h - h0,
            depth + 1,
            None,
            args,
            grid,
            st,
            prev_store,
            adaptive,
        )

        split_bits = 0.0

        if qt_ok:
            split_bits += qt_split_flag_bits(adaptive, depth, 0)

        split_bits += 1.0  # extra_split_flag = 1
        split_bits += 1.0  # extra_split_dir_flag

        cand.append(
            parent_node(
                x,
                y,
                w,
                h,
                depth,
                parent,
                "bh",
                split_bits,
                [c0, c1],
                args,
                qt_flag_present=qt_ok,
            )
        )

    # binary vertical: left/right
    if bv_ok:
        st = list(store)
        w0 = w // 2

        c0 = eval_leaf(
            padded,
            x,
            y,
            w0,
            h,
            depth + 1,
            None,
            args,
            grid,
            st,
            prev_store,
            adaptive,
        )
        add_leaves_to_store(c0, st)

        c1 = eval_leaf(
            padded,
            x + w0,
            y,
            w - w0,
            h,
            depth + 1,
            None,
            args,
            grid,
            st,
            prev_store,
            adaptive,
        )

        split_bits = 0.0

        if qt_ok:
            split_bits += qt_split_flag_bits(adaptive, depth, 0)

        split_bits += 1.0  # extra_split_flag = 1
        split_bits += 1.0  # extra_split_dir_flag

        cand.append(
            parent_node(
                x,
                y,
                w,
                h,
                depth,
                parent,
                "bv",
                split_bits,
                [c0, c1],
                args,
                qt_flag_present=qt_ok,
            )
        )

    # quad split
    if qt_ok:
        st = list(store)
        w0 = w // 2
        h0 = h // 2

        specs = [
            (x, y, w0, h0),
            (x + w0, y, w - w0, h0),
            (x, y + h0, w0, h - h0),
            (x + w0, y + h0, w - w0, h - h0),
        ]

        children = []

        for cx, cy, cw, ch in specs:
            c = encode_node(
                padded,
                cx,
                cy,
                cw,
                ch,
                depth + 1,
                None,
                args,
                grid,
                st,
                prev_store,
                adaptive,
            )

            children.append(c)
            add_leaves_to_store(c, st)

        split_bits = qt_split_flag_bits(adaptive, depth, 1)

        cand.append(
            parent_node(
                x,
                y,
                w,
                h,
                depth,
                parent,
                "qt",
                split_bits,
                children,
                args,
                qt_flag_present=True,
            )
        )

    best = min(cand, key=lambda n: n.cost)
    best.parent = parent

    return best


def commit_node(node, store, adaptive, writer, frame_idx):
    qt_split_flag_update(adaptive, node)

    if not node.is_leaf():
        for c in node.children:
            commit_node(c, store, adaptive, writer, frame_idx)
        return

    b = node.best

    if adaptive is not None:
        if len(node.avail_modes) > 1:
            adaptive["mode"].update(b.mode)

        if b.mode == "copy" and len(node.avail_cands) > 1:
            if "copy_candidate_unary" in adaptive:
                unary_candidate_update(
                    node.avail_cands.index(b.candidate_name),
                    len(node.avail_cands),
                    adaptive["copy_candidate_unary"],
                )
            else:
                adaptive["candidate"].update(b.candidate_name)

        elif b.mode == "delta" and len(node.avail_cands) > 1:
            adaptive["candidate"].update(b.candidate_name)

        if b.mode == "delta" and "delta_res_abs_a" in adaptive:
            for q, k in zip(b.q_values, "abc"):
                adaptive_signed_residual_update(
                    q,
                    adaptive[f"delta_res_abs_{k}"],
                    adaptive["delta_abs_max"],
                )

    store.append(LeafRecord(node.x, node.y, node.w, node.h, b.plane))

    if writer:
        q = list(b.q_values) + ["", "", ""]

        writer.writerow(
            {
                "frame": frame_idx,
                "bx": node.x,
                "by": node.y,
                "block_w": node.w,
                "block_h": node.h,
                "qt_depth": node.depth,
                "split_type": node.split,
                "mode": b.mode,
                "candidate": b.candidate_name,
                "bits": node.bits,
                "split_bits": node.split_bits,
                "sse": node.sse,
                "cost": node.cost,
                "q0": q[0],
                "q1": q[1],
                "q2": q[2],
                "actual_a": node.actual.a,
                "actual_b": node.actual.b,
                "actual_c": node.actual.c,
                "recon_a": b.plane.a,
                "recon_b": b.plane.b,
                "recon_c": b.plane.c,
            }
        )


def paint(node, recon):
    if node.is_leaf():
        recon[node.y : node.y + node.h, node.x : node.x + node.w] = node.best.recon_block
        return

    for c in node.children:
        paint(c, recon)


def collect(node, s):
    s["split_bits"] += node.split_bits

    if node.split == "qt":
        s["qt_nodes"] += 1
    elif node.split == "bh":
        s["bin_h_nodes"] += 1
    elif node.split == "bv":
        s["bin_v_nodes"] += 1

    if node.is_leaf():
        b = node.best
        s["leaf_blocks"] += 1
        s[f"{b.mode}_blocks"] += 1
        s[f"candidate_{b.candidate_name}_count"] = s.get(
            f"candidate_{b.candidate_name}_count",
            0,
        ) + 1

        if b.mode == "delta":
            s["delta_mode_count"] += 1

            if b.q_values == (0, 0, 0):
                s["zero_delta_blocks"] += 1

        return

    for c in node.children:
        collect(c, s)


def pad_to_block_multiple(img, bs):
    h, w = img.shape
    ph = (bs - h % bs) % bs
    pw = (bs - w % bs) % bs

    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw)), mode="edge")

    return img.copy(), h + ph, w + pw


def compute_metrics(orig, recon, maxv):
    d = orig.astype(np.float64) - recon.astype(np.float64)
    mse = float(np.mean(d * d))

    return {
        "mae": float(np.mean(np.abs(d))),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "psnr": float("inf") if mse == 0 else 10.0 * math.log10(maxv * maxv / mse),
        "max_error": float(np.max(np.abs(d))),
    }


def simulate_one_frame(depth, frame_idx, args, grid, prev_store=None, writer=None, adaptive=None):
    h, w = depth.shape
    padded, hp, wp = pad_to_block_multiple(depth, args.block_size)

    recon = np.zeros_like(padded, dtype=np.float64)
    store = []
    prev_store = prev_store or []

    root_count = 0
    total_bits = 0.0
    total_sse = 0.0

    st = {
        "leaf_blocks": 0,
        "qt_nodes": 0,
        "bin_h_nodes": 0,
        "bin_v_nodes": 0,
        "split_bits": 0.0,
        "direct_blocks": 0,
        "copy_blocks": 0,
        "delta_blocks": 0,
        "zero_delta_blocks": 0,
        "delta_mode_count": 0,
    }

    for y in range(0, hp, args.block_size):
        for x in range(0, wp, args.block_size):
            root_count += 1

            root = encode_node(
                padded,
                x,
                y,
                args.block_size,
                args.block_size,
                0,
                None,
                args,
                grid,
                store,
                prev_store,
                adaptive,
            )

            commit_node(root, store, adaptive, writer, frame_idx)
            paint(root, recon)
            collect(root, st)

            total_bits += root.bits
            total_sse += root.sse

    rec = recon[:h, :w]
    m = compute_metrics(depth, rec, args.max_value)

    leaves = max(int(st["leaf_blocks"]), 1)

    summary = {
        "frame": frame_idx,
        "width": w,
        "height": h,
        "padded_width": wp,
        "padded_height": hp,
        "root_block_size": args.block_size,
        "max_qt_depth": args.max_qt_depth,
        "num_roots": root_count,
        "leaf_blocks": int(st["leaf_blocks"]),
        "qt_nodes": int(st["qt_nodes"]),
        "bin_h_nodes": int(st["bin_h_nodes"]),
        "bin_v_nodes": int(st["bin_v_nodes"]),
        "split_bits": float(st["split_bits"]),
        "bits": total_bits,
        "bpp": total_bits / (h * w),
        "sse": total_sse,
        **m,
        "direct_blocks": int(st["direct_blocks"]),
        "copy_blocks": int(st["copy_blocks"]),
        "delta_blocks": int(st["delta_blocks"]),
        "direct_ratio": st["direct_blocks"] / leaves,
        "copy_ratio": st["copy_blocks"] / leaves,
        "delta_ratio": st["delta_blocks"] / leaves,
        "zero_delta_blocks": int(st["zero_delta_blocks"]),
        "zero_delta_ratio_in_delta": (
            st["zero_delta_blocks"] / st["delta_mode_count"]
            if st["delta_mode_count"]
            else 0.0
        ),
    }

    for k, v in st.items():
        if k.startswith("candidate_"):
            summary[k.replace("-", "_")] = int(v)

    if adaptive is not None:
        summary.update(adaptive["mode"].snapshot("final_mode"))
        summary.update(adaptive["candidate"].snapshot("final_candidate"))

        if "copy_candidate_unary" in adaptive:
            for i, c in enumerate(adaptive["copy_candidate_unary"]):
                summary[f"final_copy_unary_ctx{i}_p1"] = c.p1

        if "qt_split" in adaptive:
            for i, c in enumerate(adaptive["qt_split"]):
                summary[f"final_qt_split_depth{i}_p1"] = c.p1

        if "delta_res_abs_a" in adaptive:
            for k in "abc":
                summary.update(
                    adaptive[f"delta_res_abs_{k}"].snapshot(f"final_delta_abs_{k}")
                )

    return rec, summary, store


def read_yuv420p10le_y_frame(fp, idx, w, h):
    fp.seek(idx * w * h * 3)

    raw = fp.read(w * h * 2)

    if len(raw) != w * h * 2:
        raise EOFError(f"Failed to read frame {idx}")

    return np.frombuffer(raw, dtype="<u2").reshape(h, w).astype(np.float64)


def count_frames(path, w, h):
    return os.path.getsize(path) // (w * h * 3)


def write_yuv420p10le_frame(fp, y, w, h, maxv):
    fp.write(np.clip(np.rint(y), 0, maxv).astype("<u2").tobytes())

    uv = np.full((h // 2, w // 2), min(512, maxv), dtype="<u2")
    fp.write(uv.tobytes())
    fp.write(uv.tobytes())


def parse_args():
    p = argparse.ArgumentParser(
        description="Depth plane compression simulation with recursive split."
    )

    p.add_argument("--input", required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)

    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=0)

    p.add_argument("--block-size", type=int, default=128, help="root square block size")
    p.add_argument("--max-qt-depth", type=int, default=0, help="max recursive quad split depth")

    p.add_argument("--lambda-rd", type=float, default=0.0)
    p.add_argument("--qa", type=float, default=0.25)
    p.add_argument("--qb", type=float, default=0.25)
    p.add_argument("--qc", type=float, default=2.0)

    p.add_argument("--mode-bits", type=int, default=2)
    p.add_argument("--max-value", type=int, default=1023)

    p.add_argument("--temporal-candidate", action="store_true")

    p.add_argument("--adaptive-prob", action="store_true")
    p.add_argument("--copy-candidate-unary", action="store_true")
    p.add_argument("--qt-split-adaptive", action="store_true")

    p.add_argument("--delta-residual-adaptive", action="store_true")
    p.add_argument("--delta-abs-max", type=int, default=7)

    p.add_argument("--max-candidates", type=int, default=8)

    p.add_argument("--prob-lr", type=float, default=0.05)
    p.add_argument("--prob-min", type=float, default=0.02)
    p.add_argument("--prob-max", type=float, default=0.95)
    p.add_argument("--prob-reset", choices=["frame", "sequence"], default="frame")

    p.add_argument("--out-csv", default="depth_plane_frame_stats.csv")
    p.add_argument("--out-json", default="depth_plane_summary.json")
    p.add_argument("--out-recon-yuv", default="")
    p.add_argument("--out-block-csv", default="")

    return p.parse_args()


def main():
    args = parse_args()

    if args.block_size <= 0 or args.max_qt_depth < 0:
        raise ValueError("bad block/split size")

    if min(args.qa, args.qb, args.qc) <= 0:
        raise ValueError("qstep must be positive")

    if args.width % 2 or args.height % 2:
        raise ValueError("yuv420p10le requires even width/height")

    if args.copy_candidate_unary and not args.adaptive_prob:
        raise ValueError("--copy-candidate-unary requires --adaptive-prob")

    if args.qt_split_adaptive and not args.adaptive_prob:
        raise ValueError("--qt-split-adaptive requires --adaptive-prob")

    if args.delta_residual_adaptive and not args.adaptive_prob:
        raise ValueError("--delta-residual-adaptive requires --adaptive-prob")

    if args.max_candidates <= 0 or args.delta_abs_max < 0:
        raise ValueError("bad candidate/residual setting")

    if not (0 <= args.prob_lr <= 1):
        raise ValueError("bad probability setting")

    if args.prob_min < 0 or args.prob_max <= 0 or args.prob_min >= args.prob_max:
        raise ValueError("bad probability setting")

    total = count_frames(args.input, args.width, args.height)

    if total <= 0:
        raise ValueError("no complete frames found")

    if args.start_frame < 0 or args.start_frame >= total:
        raise ValueError("bad frame range")

    end = total if args.num_frames == 0 else min(total, args.start_frame + args.num_frames)

    grid = GridCache()

    seq_adapt = (
        create_adaptive_models(args)
        if args.adaptive_prob and args.prob_reset == "sequence"
        else None
    )

    summaries = []
    prev_store = None

    recon_fp = open(args.out_recon_yuv, "wb") if args.out_recon_yuv else None

    block_fp = None
    writer = None

    if args.out_block_csv:
        block_fp = open(args.out_block_csv, "w", newline="")

        fields = [
            "frame",
            "bx",
            "by",
            "block_w",
            "block_h",
            "qt_depth",
            "split_type",
            "mode",
            "candidate",
            "bits",
            "split_bits",
            "sse",
            "cost",
            "q0",
            "q1",
            "q2",
            "actual_a",
            "actual_b",
            "actual_c",
            "recon_a",
            "recon_b",
            "recon_c",
        ]

        writer = csv.DictWriter(block_fp, fieldnames=fields)
        writer.writeheader()

    try:
        with open(args.input, "rb") as fp:
            for fi in range(args.start_frame, end):
                if args.adaptive_prob and args.prob_reset == "frame":
                    adaptive = create_adaptive_models(args)
                else:
                    adaptive = seq_adapt

                depth = read_yuv420p10le_y_frame(fp, fi, args.width, args.height)

                rec, sm, cur_store = simulate_one_frame(
                    depth,
                    fi,
                    args,
                    grid,
                    prev_store=prev_store,
                    writer=writer,
                    adaptive=adaptive,
                )

                prev_store = cur_store
                summaries.append(sm)

                if recon_fp:
                    write_yuv420p10le_frame(
                        recon_fp,
                        rec,
                        args.width,
                        args.height,
                        args.max_value,
                    )

                print(
                    f"Frame {fi:4d} | "
                    f"bits={sm['bits']:.1f} | "
                    f"bpp={sm['bpp']:.5f} | "
                    f"leaf={sm['leaf_blocks']} | "
                    f"QT={sm['qt_nodes']} | "
                    f"BH/BV={sm['bin_h_nodes']}/{sm['bin_v_nodes']} | "
                    f"splitBits={sm['split_bits']:.1f} | "
                    f"MAE={sm['mae']:.4f} | "
                    f"PSNR={sm['psnr']:.3f} | "
                    f"D/C/Δ={sm['direct_ratio']:.3f}/"
                    f"{sm['copy_ratio']:.3f}/"
                    f"{sm['delta_ratio']:.3f}"
                )

    finally:
        if recon_fp:
            recon_fp.close()

        if block_fp:
            block_fp.close()

    if not summaries:
        raise RuntimeError("No frames processed")

    with open(args.out_csv, "w", newline="") as f:
        fields = sorted(set().union(*(s.keys() for s in summaries)))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summaries)

    avg = {}

    for k in sorted(set().union(*(s.keys() for s in summaries))):
        vals = []

        for s in summaries:
            try:
                v = float(s[k])

                if math.isfinite(v):
                    vals.append(v)
            except Exception:
                pass

        if vals:
            avg[k] = float(np.mean(vals))

    total_bits = float(sum(s["bits"] for s in summaries))
    total_pixels = float(args.width * args.height * len(summaries))

    overall = {
        **vars(args),
        "num_processed_frames": len(summaries),
        "total_bits": total_bits,
        "overall_bpp": total_bits / total_pixels,
        "average": avg,
        "frame_csv": args.out_csv,
        "recon_yuv": args.out_recon_yuv,
        "block_csv": args.out_block_csv,
    }

    with open(args.out_json, "w") as f:
        json.dump(overall, f, indent=2)

    print()
    print("Done.")
    print(f"Frame CSV : {args.out_csv}")
    print(f"Summary   : {args.out_json}")

    if args.out_recon_yuv:
        print(f"Recon YUV : {args.out_recon_yuv}")

    if args.out_block_csv:
        print(f"Block CSV : {args.out_block_csv}")

    print(f"Overall bpp: {overall['overall_bpp']:.6f}")


if __name__ == "__main__":
    main()
