from __future__ import annotations

import math
import threading
import time
import traceback
import os
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable

import networkx as nx
import torch
import multiprocessing as mp
import tqdm
import logging
from .TimingGraph import TimingGraph
from .Data_var import chunk_dir, ep_dir
from utils.logger import setup_logging
from utils.env_setup import setup_env

setup_env()
logger = logging.getLogger("Chunk_Store")
setup_logging(logger, "INFO")


def _safe4(x) -> List[float]:
    if x is None:
        return [0.0, 0.0, 0.0, 0.0]
    out = []
    for i in range(4):
        v = x[i] if i < len(x) else None
        if v is None:
            out.append(0.0)
        else:
            f = float(v)
            out.append(f if math.isfinite(f) else 0.0)
    return out


def _safe_bool(x) -> int:
    return int(bool(x))


def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    """Append one JSON object to a .jsonl file, creating parent dirs if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def annotate_outpin_chunks_with_membership(
    *,
    chunk_dir: str,
    membership_jsonl: str,
    meta_key: str = "cone_memberships",
    max_memberships_per_chunk: Optional[int] = None,
    log_every: int = 2000,
) -> None:
    """
    Optional post-process:
      - Read membership_jsonl rows: {design_id,outpin_name,cone_signature,ep_name}
      - For each (design_id,outpin_name), locate the 3-hop chunk signature via chunk_dir/index.jsonl
      - Load chunk payload, add meta[meta_key] as list[dict], save back.

    Compatibility:
      - Only adds a new meta field; existing training code that ignores unknown meta keys will keep working.
    """
    layout = ChunkLayout(chunk_dir)
    if not os.path.exists(layout.index_path):
        logger.warning(f"[annotate] index.jsonl not found: {layout.index_path}")
        return

    # Build (design_id, center_name)->signature map for 3-hop chunks
    center_map: Dict[Tuple[str, str], str] = {}
    with open(layout.index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            meta = r.get("meta", {}) or {}
            sig = r.get("signature")
            if not sig:
                continue
            did = meta.get("design_id")
            cname = meta.get("center_name")
            if did is None or cname is None:
                continue
            center_map[(str(did), str(cname))] = str(sig)

    # Aggregate memberships per outpin
    agg: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    if not os.path.exists(membership_jsonl):
        logger.warning(f"[annotate] membership_jsonl not found: {membership_jsonl}")
        return

    with open(membership_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            did = str(r.get("design_id"))
            op = str(r.get("outpin_name"))
            key = (did, op)
            agg.setdefault(key, []).append(
                {
                    "cone_signature": r.get("cone_signature"),
                    "ep_name": r.get("ep_name"),
                }
            )

    done = 0
    for (did, op), mem_list in agg.items():
        sig = center_map.get((did, op), None)
        if sig is None:
            continue
        p = layout.chunk_path(sig)
        payload = torch.load(p, map_location="cpu")
        meta = payload.get("meta", {}) or {}
        old = meta.get(meta_key, [])
        if not isinstance(old, list):
            old = []
        new = old + mem_list
        if max_memberships_per_chunk is not None:
            new = new[: int(max_memberships_per_chunk)]
        meta[meta_key] = new
        payload["meta"] = meta
        torch.save(payload, p)

        done += 1
        if log_every > 0 and (done % log_every) == 0:
            logger.info(f"[annotate] updated {done} outpin chunks ...")

    logger.info(f"[annotate] done. updated outpin chunks = {done}")



# -----------------------------
# Keys & Meta (now pin-centered)
# -----------------------------

@dataclass(frozen=True)
class StorageKey:
    """
    Storage key uniquely identifies one stored chunk.

    Args:
        design_id: Design identifier.
        center_type: Center type string. For this exporter: "outpin".
        center_name: Center node name in TG.G, e.g., "U222/Z".
    """
    design_id: str
    center_type: str
    center_name: str


@dataclass
class ChunkMeta:
    """
    ChunkMeta is a class that stores the metadata of a chunk.
    Args:
        design_id: The design id.
        center_type: The center type.
        center_name: The center name.
        k_hop: The k-hop radius.
        num_nodes: The number of nodes in the chunk.
        num_edges: The number of edges in the chunk.
    """
    design_id: str
    center_type: str
    center_name: str
    k_hop: int
    num_nodes: int
    num_edges: int

class _Heartbeat:
    """Periodic heartbeat logger. Set ``enabled=False`` to make start/stop no-ops."""

    def __init__(self, logger, prefix: str, interval_s: int = 10, status_fn=None, enabled: bool = True):
        self.logger = logger
        self.prefix = prefix
        self.interval_s = interval_s
        self.status_fn = status_fn
        self.enabled = enabled
        self._stop = threading.Event()
        self._t0 = None
        self._th = None

    def start(self):
        if not self.enabled:
            return
        self._t0 = time.time()

        def _run():
            i = 0
            while not self._stop.wait(self.interval_s):
                i += 1
                dt = time.time() - self._t0
                extra = ""
                if self.status_fn is not None:
                    try:
                        extra = " | " + str(self.status_fn())
                    except Exception:
                        extra = ""
                self.logger.info(f"{self.prefix} still running... elapsed={dt:.1f}s (tick={i}){extra}")

        self._th = threading.Thread(target=_run, daemon=True)
        self._th.start()

    def stop(self):
        if not self.enabled:
            return
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=1.0)


# -----------------------------
# PackedTimingGraph: add outpin centers list (directed only)
# -----------------------------

class PackedTimingGraph:
    """
    PackedTimingGraph is a class that packs a TimingGraph into a format that can be used by the ChunkDiskWriter.
    Args:
        G: The TimingGraph to pack.
        design_id: The design id.
    """
    def __init__(self, G: nx.MultiDiGraph, *, design_id: str):
        self.design_id = design_id
        self.G = G

        self.node_names: List[str] = []
        self.name2nid: Dict[str, int] = {}
        self.adj: List[List[int]] = []  # directed adjacency (out-neighbors)
        self.adj_in: List[List[int]] = []  # directed in-neighbors
        self.adj_u: List[List[int]] = []  # undirected adjacency for k-hop
        self.edges: List[Tuple[int, int, int, Dict[str, Any]]] = []  # src, dst, key, features
        self.out_edges: List[List[int]] = []  # outgoing edge indices per src nid
        self.node_feat: Dict[str, torch.Tensor] = {}

        # list of outpin node ids
        self.outpin_nids: List[int] = []

    def pack(self) -> "PackedTimingGraph":
        self.node_names = list(self.G.nodes())
        self.name2nid = {n: i for i, n in enumerate(self.node_names)}
        N = len(self.node_names)
        t0 = time.time()

        # --- batch node feature extraction (no per-node tensor creation) ---
        _s4_keys = ("slack", "slack_eco", "arrival", "trans", "ceff", "bbox")
        buf_f = {k: [] for k in _s4_keys}
        buf_level, buf_is_port, buf_is_outpin = [], [], []
        buf_clock, buf_async, buf_type, buf_size, buf_size_eco, buf_crit = [], [], [], [], [], []
        outpin_nids: List[int] = []

        for i, n in enumerate(self.node_names):
            nd = self.G.nodes[n]
            for k in _s4_keys:
                buf_f[k].append(_safe4(nd.get(k)))
            buf_level.append(int(nd.get("level", 0) or 0))
            buf_is_port.append(_safe_bool(nd.get("is_port", False)))
            _op = _safe_bool(nd.get("is_outpin", False))
            buf_is_outpin.append(_op)
            if _op:
                outpin_nids.append(i)
            buf_clock.append(_safe_bool(nd.get("is_clock_network", False)))
            buf_async.append(_safe_bool(nd.get("is_async_pin", False)))
            buf_type.append(int(nd.get("type_id", -1) or -1))
            buf_size.append(int(nd.get("size_id", -1) or -1))
            buf_size_eco.append(int(nd.get("size_id_eco", -1) or -1))
            buf_crit.append(_safe_bool(nd.get("criticality", False)))

        self.node_feat = {k: torch.tensor(buf_f[k], dtype=torch.float32) for k in _s4_keys}
        self.node_feat["level"] = torch.tensor(buf_level, dtype=torch.int32)
        self.node_feat["is_port"] = torch.tensor(buf_is_port, dtype=torch.int32)
        self.node_feat["is_outpin"] = torch.tensor(buf_is_outpin, dtype=torch.int32)
        self.node_feat["is_clock_network"] = torch.tensor(buf_clock, dtype=torch.int32)
        self.node_feat["is_async_pin"] = torch.tensor(buf_async, dtype=torch.int32)
        self.node_feat["type_id"] = torch.tensor(buf_type, dtype=torch.int32)
        self.node_feat["size_id"] = torch.tensor(buf_size, dtype=torch.int32)
        self.node_feat["size_id_eco"] = torch.tensor(buf_size_eco, dtype=torch.int32)
        self.node_feat["criticality"] = torch.tensor(buf_crit, dtype=torch.int32)
        self.outpin_nids = outpin_nids

        logger.info(f"[pack] {N} nodes packed in {time.time()-t0:.2f}s")

        # --- edge packing (DIRECTED only) ---
        t1 = time.time()
        adj = [[] for _ in range(N)]
        adj_in = [[] for _ in range(N)]
        adj_u = [[] for _ in range(N)]
        edges: List[Tuple[int, int, int, Dict[str, Any]]] = []
        out_edges: List[List[int]] = [[] for _ in range(N)]

        sense_map = {
            "positive_unate": 1,
            "negative_unate": 2,
            "rising_edge": 3,
            "falling_edge": 4,
            "unknown": 0,
        }

        for (u, v, k, ed) in self.G.edges(keys=True, data=True):
            su = self.name2nid[u]
            sv = self.name2nid[v]

            adj[su].append(sv)
            adj_in[sv].append(su)
            adj_u[su].append(sv)
            adj_u[sv].append(su)

            delay = _safe4(ed.get("delay"))
            is_cell = _safe_bool(ed.get("is_cell", False))
            sense = sense_map.get(ed.get("sense_unate", "unknown"), 0)

            idx = len(edges)
            edges.append((su, sv, int(k), {"is_cell": is_cell, "delay": delay, "sense_unate": int(sense)}))
            out_edges[su].append(idx)

        self.adj = adj
        self.adj_in = adj_in
        self.adj_u = adj_u
        self.edges = edges
        self.out_edges = out_edges

        logger.info(f"[pack] {len(edges)} edges packed in {time.time()-t1:.2f}s")
        return self


# -----------------------------
# Disk layout
# -----------------------------

class ChunkLayout:
    """
    Directory layout:
      out_dir/
        index.jsonl
        chunks/ab/<signature>.pt     (bucket = signature[:2])
    """
    def __init__(self, out_dir: str = chunk_dir):
        self.out_dir = out_dir
        self.index_path = os.path.join(out_dir, "index.jsonl")
        self.chunk_root = os.path.join(out_dir, "chunks")
        os.makedirs(self.chunk_root, exist_ok=True)
        if not os.path.exists(self.index_path):
            open(self.index_path, "w", encoding="utf-8").close()
        self._bucket_cache: set = set()

    def chunk_path(self, signature: str) -> str:
        bucket = signature[:2]
        if bucket not in self._bucket_cache:
            os.makedirs(os.path.join(self.chunk_root, bucket), exist_ok=True)
            self._bucket_cache.add(bucket)
        return os.path.join(self.chunk_root, bucket, f"{signature}.pt")

    def relpath(self, abspath: str) -> str:
        return os.path.relpath(abspath, self.out_dir)


# -----------------------------
# Fast k-hop + worker
# -----------------------------

def _k_hop_nodes_fast(adj: List[List[int]], seeds: List[int], k: int, fanout_cap: Optional[int]) -> List[int]:
    visited = set(seeds)
    frontier = seeds
    for _ in range(k):
        nxt_set = set()
        for u in frontier:
            nbrs = adj[u]
            if fanout_cap is not None and len(nbrs) > fanout_cap:
                nbrs = nbrs[:fanout_cap]
            for v in nbrs:
                if v not in visited:
                    nxt_set.add(v)
        frontier = list(nxt_set)
        visited |= nxt_set
        if not frontier:
            break
    return list(visited)

# -----------------------------
# Fast logic cone + worker
# -----------------------------

def _logic_cone_nodes_fast(
    adj_dir: List[List[int]],
    seed: int,
    *,
    max_hops: Optional[int] = None,      # None => no hop limit
    max_nodes: Optional[int] = 100000,     # prevent explosion
    fan_cap: Optional[int] = None,       # cap for each node's fanin/fanout
) -> List[int]:
    """
    Generic cone expansion on a directed adjacency list.
    - fanin cone: pass packed.adj_in
    - fanout cone: pass packed.adj
    """
    visited = {seed}
    # frontier holds (node, depth)
    stack = [(seed, 0)]
    while stack:
        u, d = stack.pop()
        if max_hops is not None and d >= max_hops:
            continue

        nbrs = adj_dir[u]
        if fan_cap is not None and len(nbrs) > fan_cap:
            nbrs = nbrs[:fan_cap]

        nd = d + 1
        for v in nbrs:
            if v in visited:
                continue
            visited.add(v)
            if max_nodes is not None and len(visited) >= max_nodes:
                return list(visited)
            stack.append((v, nd))
    return list(visited)

def _filter_nodes_fast(
    node_ids: List[int],
    is_clock_network: torch.Tensor,
    is_async_pin: torch.Tensor,
    *,
    exclude_clock_network: bool,
    exclude_async_pin: bool,
) -> List[int]:
    if not exclude_clock_network and not exclude_async_pin:
        return node_ids
    out = []
    for nid in node_ids:
        if exclude_clock_network and int(is_clock_network[nid].item()) == 1:
            continue
        if exclude_async_pin and int(is_async_pin[nid].item()) == 1:
            continue
        out.append(nid)
    return out


def _compute_signature(storage_key: StorageKey, k_hop: int, node_ids_sorted: List[int]) -> str:
    raw = json.dumps(
        {
            "design_id": storage_key.design_id,
            "center_type": storage_key.center_type,
            "center_name": storage_key.center_name,
            "k_hop": k_hop,
            "node_ids": node_ids_sorted,
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha1_bytes(raw)


_GLOBAL = {}

# Threshold: designs with more centers than this get heartbeat + verbose worker init
_LARGE_DESIGN_THRESHOLD = 500


class _IndexWriter:
    """Buffered writer for index.jsonl to avoid per-chunk file open/close."""

    def __init__(self, path: str, flush_every: int = 256):
        self._path = path
        self._flush_every = flush_every
        self._buf: List[str] = []

    def append(self, sig: str, rel: str, meta: Dict[str, Any]) -> None:
        row = json.dumps({"signature": sig, "path": rel, "meta": meta}, ensure_ascii=False)
        self._buf.append(row)
        if len(self._buf) >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        with open(self._path, "a", encoding="utf-8") as f:
            f.write("\n".join(self._buf) + "\n")
        self._buf.clear()

    def close(self) -> None:
        self.flush()


def _init_worker(packed: PackedTimingGraph, out_dir: str, write_node_names: bool,
                  init_counter=None, init_lock=None, verbose: bool = False):
    _GLOBAL["packed"] = packed
    _GLOBAL["layout"] = ChunkLayout(out_dir)
    _GLOBAL["write_node_names"] = write_node_names
    _GLOBAL["cone_cache"] = {}

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass

    if init_counter is not None:
        try:
            if init_lock is not None:
                with init_lock:
                    init_counter.value += 1
            else:
                init_counter.value += 1
        except Exception:
            pass

    if verbose:
        logger.info(f"[worker {os.getpid()}] init done")



def _process_one_outpin_to_disk(args) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """
    Build one outpin-centered chunk and write to disk.

    Returns (small object only):
        (signature, relpath, meta_dict)
    """
    (center_nid, k_hop, exclude_clock_network, exclude_async_pin, fanout_cap) = args
    packed: PackedTimingGraph = _GLOBAL["packed"]
    layout: ChunkLayout = _GLOBAL["layout"]
    write_node_names: bool = _GLOBAL["write_node_names"]

    try:
        # k-hop
        node_ids = _k_hop_nodes_fast(packed.adj_u, [center_nid], k_hop, fanout_cap)
        node_ids = _filter_nodes_fast(
            node_ids,
            packed.node_feat["is_clock_network"],
            packed.node_feat["is_async_pin"],
            exclude_clock_network=exclude_clock_network,
            exclude_async_pin=exclude_async_pin,
        )
        node_ids_sorted = sorted(node_ids)
        node_set = set(node_ids_sorted)
        local_id = {nid: i for i, nid in enumerate(node_ids_sorted)}

        # edges inside chunk
        edge_src_local: List[int] = []
        edge_dst_local: List[int] = []
        e_is_cell: List[int] = []
        e_delay: List[List[float]] = []
        e_sense: List[int] = []

        for su in node_ids_sorted:
            for eidx in packed.out_edges[su]:
                src, dst, _, ef = packed.edges[eidx]
                if dst in node_set:
                    edge_src_local.append(local_id[src])
                    edge_dst_local.append(local_id[dst])
                    e_is_cell.append(int(ef["is_cell"]))
                    e_delay.append(ef["delay"])
                    e_sense.append(int(ef["sense_unate"]))

        nf = {k: v[node_ids_sorted].contiguous() for k, v in packed.node_feat.items()}

        # Record all outpin names within this cone. This enables cone->outpin->3hop pairing,
        # and supports the case where one outpin appears in multiple cones.
        outpin_names_in_cone: List[str] = []
        try:
            is_outpin_full = packed.node_feat.get("is_outpin", None)
            if is_outpin_full is not None:
                for nid in node_ids_sorted:
                    if int(is_outpin_full[nid].item()) == 1:
                        outpin_names_in_cone.append(str(packed.node_names[nid]))
        except Exception:
            # Best-effort; do not fail export if outpin info is missing.
            outpin_names_in_cone = []
        ef = {
            "is_cell": torch.tensor(e_is_cell, dtype=torch.int32),
            "delay": torch.tensor(e_delay, dtype=torch.float32) if e_delay else torch.zeros((0, 4), dtype=torch.float32),
            "sense_unate": torch.tensor(e_sense, dtype=torch.int32),
        }

        center_name = str(packed.node_names[center_nid])
        storage_key = StorageKey(design_id=packed.design_id, center_type="outpin", center_name=center_name)
        signature = _compute_signature(storage_key, k_hop, node_ids_sorted)

        p = layout.chunk_path(signature)
        meta = ChunkMeta(
            design_id=storage_key.design_id,
            center_type=storage_key.center_type,
            center_name=storage_key.center_name,
            k_hop=k_hop,
            num_nodes=len(node_ids_sorted),
            num_edges=len(edge_src_local),
        )
        # Extend meta with cone-specific information while keeping backward compatibility.
        meta_dict = meta.__dict__.copy()
        meta_dict["outpins_in_cone"] = outpin_names_in_cone



        payload = {
            "signature": signature,
            "storage_key": storage_key.__dict__,
            "meta": meta_dict,
            "node_ids": torch.tensor(node_ids_sorted, dtype=torch.int64),
            "edge_src": torch.tensor(edge_src_local, dtype=torch.int64),
            "edge_dst": torch.tensor(edge_dst_local, dtype=torch.int64),
            "node_feat": nf,
            "edge_feat": ef,
        }
        if write_node_names:
            payload["node_names"] = [str(packed.node_names[nid]) for nid in node_ids_sorted]

        torch.save(payload, p)

        rel = layout.relpath(p)
        return signature, rel, meta_dict

    except Exception as e:
        center_name = None
        try:
            center_name = str(packed.node_names[center_nid])
        except Exception:
            center_name = f"<nid={center_nid}>"
        err = {
            "center_name": center_name,
            "error": repr(e),
            "traceback": traceback.format_exc(limit=20),
        }
        return "__ERROR__", "", err


def _process_one_ep_cone_to_disk(args) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """
    Build one EP-centered logic cone chunk and write to disk.
    Returns: (signature, relpath, meta_dict)
    """
    (
        center_nid,
        cone_dir,              # "fanin" or "fanout"
        max_hops,
        max_nodes,
        fan_cap,
        exclude_clock_network,
        exclude_async_pin,
    ) = args

    packed: PackedTimingGraph = _GLOBAL["packed"]
    layout: ChunkLayout = _GLOBAL["layout"]
    write_node_names: bool = _GLOBAL["write_node_names"]
    cone_cache: Dict[Tuple, List[int]] = _GLOBAL["cone_cache"]

    try:
        cache_key = (center_nid, cone_dir, max_hops, max_nodes, fan_cap, exclude_clock_network, exclude_async_pin)
        node_ids = cone_cache.get(cache_key, None)
        if node_ids is None:
            adj_dir = packed.adj_in if cone_dir == "fanin" else packed.adj
            node_ids = _logic_cone_nodes_fast(
                adj_dir,
                center_nid,
                max_hops=max_hops,
                max_nodes=max_nodes,
                fan_cap=fan_cap,
            )
            node_ids = _filter_nodes_fast(
                node_ids,
                packed.node_feat["is_clock_network"],
                packed.node_feat["is_async_pin"],
                exclude_clock_network=exclude_clock_network,
                exclude_async_pin=exclude_async_pin,
            )
            cone_cache[cache_key] = node_ids

        node_ids_sorted = sorted(node_ids)
        node_set = set(node_ids_sorted)
        local_id = {nid: i for i, nid in enumerate(node_ids_sorted)}

        edge_src_local, edge_dst_local = [], []
        e_is_cell, e_delay, e_sense = [], [], []
        for su in node_ids_sorted:
            for eidx in packed.out_edges[su]:
                src, dst, _, ef = packed.edges[eidx]
                if dst in node_set:
                    edge_src_local.append(local_id[src])
                    edge_dst_local.append(local_id[dst])
                    e_is_cell.append(int(ef["is_cell"]))
                    e_delay.append(ef["delay"])
                    e_sense.append(int(ef["sense_unate"]))

        nf = {k: v[node_ids_sorted].contiguous() for k, v in packed.node_feat.items()}
        # ------------------------------------------------------------
        # Build outpin_nids (LOCAL node ids inside this cone subgraph).
        # Order must be stable and align with outpins_in_cone:
        # we follow node_ids_sorted scan order (same as existing code path
        # that records outpins_in_cone in other exporters).
        # ------------------------------------------------------------
        outpin_names_in_cone: List[str] = []
        outpin_nids_local: List[int] = []

        is_outpin_full = packed.node_feat.get("is_outpin", None)
        if is_outpin_full is None:
            raise RuntimeError("packed.node_feat missing key 'is_outpin'; cannot build outpin_nids for distillation.")

        for nid in node_ids_sorted:  # nid is GLOBAL nid, scan in sorted order
            if int(is_outpin_full[nid].item()) == 1:
                outpin_names_in_cone.append(str(packed.node_names[nid]))
                outpin_nids_local.append(local_id[nid])  # convert GLOBAL nid -> LOCAL nid
        
        ef = {
            "is_cell": torch.tensor(e_is_cell, dtype=torch.int32),
            "delay": torch.tensor(e_delay, dtype=torch.float32) if e_delay else torch.zeros((0, 4), dtype=torch.float32),
            "sense_unate": torch.tensor(e_sense, dtype=torch.int32),
        }

        center_name = str(packed.node_names[center_nid])

        storage_key = StorageKey(design_id=packed.design_id, center_type="ep", center_name=center_name)

        raw = json.dumps(
            {
                "design_id": storage_key.design_id,
                "center_type": storage_key.center_type,
                "center_name": storage_key.center_name,
                "cone_dir": cone_dir,
                "max_hops": max_hops,
                "max_nodes": max_nodes,
                "fan_cap": fan_cap,
                "node_ids": node_ids_sorted,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        signature = _sha1_bytes(raw)

        p = layout.chunk_path(signature)
        meta = ChunkMeta(
            design_id=storage_key.design_id,
            center_type=storage_key.center_type,
            center_name=storage_key.center_name,
            k_hop=-1,  # cone is not k-hop; but use -1 to keep dataclass
            num_nodes=len(node_ids_sorted),
            num_edges=len(edge_src_local),
        )

        meta_dict = meta.__dict__.copy()
        meta_dict["outpins_in_cone"] = outpin_names_in_cone  # for pairing/debug

        payload = {
            "signature": signature,
            "storage_key": storage_key.__dict__,
            "meta": meta_dict,
            "cone": {
                "dir": cone_dir,
                "max_hops": max_hops,
                "max_nodes": max_nodes,
                "fan_cap": fan_cap,
            },
            "node_ids": torch.tensor(node_ids_sorted, dtype=torch.int64),
            "edge_src": torch.tensor(edge_src_local, dtype=torch.int64),
            "edge_dst": torch.tensor(edge_dst_local, dtype=torch.int64),
            "node_feat": nf,
            "edge_feat": ef,

            # IMPORTANT: LOCAL nids inside this cone graph (used by teacher distill)
            "outpin_nids": torch.tensor(outpin_nids_local, dtype=torch.int64),
        }
        if write_node_names:
            payload["node_names"] = [str(packed.node_names[nid]) for nid in node_ids_sorted]

        torch.save(payload, p)
        rel = layout.relpath(p)
        return signature, rel, meta_dict

    except Exception as e:
        center_name = None
        try:
            center_name = str(packed.node_names[center_nid])
        except Exception:
            center_name = f"<nid={center_nid}>"
        err = {
            "center_name": center_name,
            "error": repr(e),
            "traceback": traceback.format_exc(limit=20),
        }
        return "__ERROR__", "", err

# -----------------------------
# Exporter: main process only writes index.jsonl
# -----------------------------

class OutpinChunkExporter:
    """
    Export outpin-centered k-hop chunks from TimingGraph to disk.

    Args:
        TG: TimingGraph instance (expects TG.G to be nx.MultiDiGraph).
        out_dir: Output directory.
        design_id: Optional override.
        packed: Optional pre-built PackedTimingGraph to skip packing.
    """

    def __init__(self, TG: Any, *, out_dir: str = chunk_dir, design_id: Optional[str] = None,
                 packed: Optional["PackedTimingGraph"] = None):
        self.TG = TG
        self.G: nx.MultiDiGraph = TG.G
        self.design_id = design_id or getattr(TG, "design", None) or "unknown_design"
        self.out_dir = out_dir
        self._packed = packed

    def export(
        self,
        *,
        k_hop: int = 3,
        exclude_clock_network: bool = False,
        exclude_async_pin: bool = False,
        fanout_cap: Optional[int] = None,
        num_workers: Optional[int] = None,
        write_node_names: bool = True,
        outpin_filter: Optional[Iterable[str]] = None,
        ep_names: Optional[Iterable[str]] = None,
        chunksize: int = 128,
        save_norm_stats: bool = False,
        norm_stats_signatures: Optional[List[str]] = None,
        mp_start_method: str = "forkserver",
        log_every: int = 2000,
    ) -> str:
        """
        Outputs:
          - out_dir/index.jsonl
          - out_dir/chunks/<bucket>/<signature>.pt

        Note:
          - If `ep_names` is provided, only outpins that fall inside the fanin logic cones of the given endpoints
            will be chunked and stored (this is applied after `outpin_filter`).

        """
        os.makedirs(self.out_dir, exist_ok=True)
        layout = ChunkLayout(self.out_dir)

        # pack once (DIRECTED) — reuse if caller already built it
        if self._packed is not None:
            packed = self._packed
        else:
            packed = PackedTimingGraph(self.G, design_id=self.design_id).pack()

        # choose centers
        if outpin_filter is None:
            centers = packed.outpin_nids
        else:
            centers = [packed.name2nid[n] for n in outpin_filter if n in packed.name2nid]
            centers = [nid for nid in centers if int(packed.node_feat["is_outpin"][nid].item()) == 1]

        # OPTIONAL: restrict outpin centers to those inside fanin logic cones of given endpoints
        if ep_names is not None:
            ep_nids: List[int] = []
            for n in ep_names:
                if n in packed.name2nid:
                    ep_nids.append(packed.name2nid[n])

            # build allowed nid set as union of fanin-cone nodes of each endpoint
            allowed: set[int] = set()
            for ep_nid in ep_nids:
                # defaults aligned with EP cone exporter usage (unbounded hops; large cap to avoid explosion)
                cone_nodes = _logic_cone_nodes_fast(
                    packed.adj_in,
                    ep_nid,
                    max_hops=None,
                    max_nodes=100000,
                    fan_cap=None,
                )
                allowed.update(cone_nodes)

            centers = [nid for nid in centers if nid in allowed]

        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)

        total = len(centers)
        is_large = total >= _LARGE_DESIGN_THRESHOLD
        idx_w = _IndexWriter(layout.index_path)

        # single-process fast path
        if num_workers <= 1:
            _init_worker(packed, self.out_dir, write_node_names)
            for nid in centers:
                r = _process_one_outpin_to_disk((nid, k_hop, exclude_clock_network, exclude_async_pin, fanout_cap))
                if r is None:
                    continue
                sig, rel, meta = r
                idx_w.append(sig, rel, meta)
            idx_w.close()
            return self.out_dir

        # multiprocess
        try:
            ctx = mp.get_context(mp_start_method)
        except ValueError:
            logger.warning(f"Unknown mp_start_method={mp_start_method}, fallback to 'fork'")
            ctx = mp.get_context("fork")

        logger.info(
            f"Start exporting chunks (centers={total}, workers={num_workers}, "
            f"chunksize={chunksize}, start_method={ctx.get_start_method()})"
        )

        init_counter, init_lock = None, None
        if is_large:
            mgr = ctx.Manager()
            init_counter = mgr.Value("i", 0)
            init_lock = mgr.Lock()

        hb = _Heartbeat(
            logger,
            prefix="[OutpinChunk Pool]",
            interval_s=15,
            status_fn=(lambda: f"workers_inited={int(init_counter.value)}/{num_workers}") if is_large else None,
            enabled=is_large,
        )

        hb.start()
        t0 = time.time()
        done = 0
        err_cnt = 0

        try:
            with ctx.Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=(packed, self.out_dir, write_node_names, init_counter, init_lock, is_large),
            ) as pool:
                hb.stop()
                if is_large:
                    logger.info("Pool created. Start scheduling jobs ...")

                it = pool.imap_unordered(
                    _process_one_outpin_to_disk,
                    ((nid, k_hop, exclude_clock_network, exclude_async_pin, fanout_cap) for nid in centers),
                    chunksize=chunksize,
                )

                for r in tqdm.tqdm(it, total=total, desc="Exporting chunks", colour="cyan",
                                   mininterval=0.5 if is_large else 2.0):
                    if r is None:
                        continue

                    sig, rel, meta = r

                    if sig == "__ERROR__":
                        err_cnt += 1
                        logger.error(
                            f"[worker error #{err_cnt}] center={meta.get('center_name')} err={meta.get('error')}\n"
                            f"{meta.get('traceback','')}"
                        )
                        continue

                    idx_w.append(sig, rel, meta)
                    done += 1

                    if log_every > 0 and (done % log_every) == 0:
                        dt = time.time() - t0
                        speed = done / max(dt, 1e-6)
                        logger.info(f"Progress: {done}/{total} chunks written, errors={err_cnt}, speed={speed:.2f} chunk/s")
        finally:
            hb.stop()
            idx_w.close()

        logger.info(f"Export finished: written={done}, errors={err_cnt}, total_centers={total}, elapsed={time.time()-t0:.1f}s")

        if save_norm_stats:
            compute_and_save_norm_stats(
                out_dir=self.out_dir,
                signatures=norm_stats_signatures,
            )

        return self.out_dir

class EndpointLogicConeExporter:
    def __init__(self, TG: Any, *, out_dir: str = chunk_dir, design_id: Optional[str] = None,
                 packed: Optional["PackedTimingGraph"] = None):
        self.TG = TG
        self.G: nx.MultiDiGraph = TG.G
        self.design_id = design_id or getattr(TG, "design", None) or "unknown_design"
        self.out_dir = out_dir
        self._packed = packed

    def export(
        self,
        *,
        endpoints: Iterable[str],          # provided EP names (node names in TG.G)
        cone_dir: str = "fanin",           # fanin/fanout
        max_hops: Optional[int] = None,
        max_nodes: int = 5000,
        fan_cap: Optional[int] = None,
        exclude_clock_network: bool = False,
        exclude_async_pin: bool = False,
        num_workers: Optional[int] = None,
        write_node_names: bool = True,
        chunksize: int = 128,
        mp_start_method: str = "forkserver",
        log_every: int = 2000,
        cone_membership_out: Optional[str] = os.path.join(chunk_dir, "cone_membership.jsonl"),
        annotate_outpin_chunks: bool = False,
        outpin_chunk_dir: Optional[str] = chunk_dir,
    ) -> str:
        os.makedirs(self.out_dir, exist_ok=True)
        layout = ChunkLayout(self.out_dir)

        # Optional: record cone membership for each outpin (many-to-many: one outpin can belong to multiple cones).
        if cone_membership_out is not None:
            os.makedirs(os.path.dirname(cone_membership_out) or ".", exist_ok=True)
            # Start fresh for this export call
            try:
                if os.path.exists(cone_membership_out):
                    os.remove(cone_membership_out)
            except Exception:
                pass


        # reuse if caller already built it
        if self._packed is not None:
            packed = self._packed
        else:
            packed = PackedTimingGraph(self.G, design_id=self.design_id).pack()

        centers = []
        for n in endpoints:
            if n in packed.name2nid:
                centers.append(packed.name2nid[n])

        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)

        total = len(centers)
        is_large = total >= _LARGE_DESIGN_THRESHOLD
        idx_w = _IndexWriter(layout.index_path)
        membership_buf: List[str] = []  # buffered membership rows

        def _flush_membership():
            if not membership_buf or cone_membership_out is None:
                return
            with open(cone_membership_out, "a", encoding="utf-8") as f:
                f.write("\n".join(membership_buf) + "\n")
            membership_buf.clear()

        def _collect_membership(sig: str, meta: Dict[str, Any]):
            if cone_membership_out is None:
                return
            ep_name = meta.get("center_name")
            for op in (meta.get("outpins_in_cone") or []):
                membership_buf.append(json.dumps({
                    "design_id": self.design_id,
                    "outpin_name": str(op),
                    "cone_signature": sig,
                    "ep_name": ep_name,
                }, ensure_ascii=False))
            if len(membership_buf) >= 256:
                _flush_membership()

        # single-process fast path
        if num_workers <= 1:
            _init_worker(packed, self.out_dir, write_node_names)
            for nid in centers:
                r = _process_one_ep_cone_to_disk(
                    (nid, cone_dir, max_hops, max_nodes, fan_cap, exclude_clock_network, exclude_async_pin)
                )
                if r is None:
                    continue
                sig, rel, meta = r
                idx_w.append(sig, rel, meta)
                _collect_membership(sig, meta)
            idx_w.close()
            _flush_membership()
            return self.out_dir

        # multiprocess
        try:
            ctx = mp.get_context(mp_start_method)
        except ValueError:
            logger.warning(f"Unknown mp_start_method={mp_start_method}, fallback to 'fork'")
            ctx = mp.get_context("fork")

        t0 = time.time()
        done, err_cnt = 0, 0

        init_counter, init_lock = None, None
        if is_large:
            mgr = ctx.Manager()
            init_counter = mgr.Value("i", 0)
            init_lock = mgr.Lock()

        hb = _Heartbeat(
            logger,
            prefix="[EPCone Pool]",
            interval_s=15,
            status_fn=(lambda: f"workers_inited={int(init_counter.value)}/{num_workers}") if is_large else None,
            enabled=is_large,
        )

        if is_large:
            logger.info(f"Creating pool (centers={total}, workers={num_workers}) ...")
        hb.start()
        try:
            with ctx.Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=(packed, self.out_dir, write_node_names, init_counter, init_lock, is_large),
            ) as pool:
                hb.stop()
                it = pool.imap_unordered(
                    _process_one_ep_cone_to_disk,
                    (
                        (nid, cone_dir, max_hops, max_nodes, fan_cap, exclude_clock_network, exclude_async_pin)
                        for nid in centers
                    ),
                    chunksize=chunksize,
                )
                for r in tqdm.tqdm(it, total=total, desc="Exporting EP cones", colour="cyan",
                                   mininterval=0.5 if is_large else 2.0):
                    if r is None:
                        continue
                    sig, rel, meta = r
                    if sig == "__ERROR__":
                        err_cnt += 1
                        logger.error(
                            f"[worker error #{err_cnt}] center={meta.get('center_name')} err={meta.get('error')}\n"
                            f"{meta.get('traceback','')}"
                        )
                        continue
                    idx_w.append(sig, rel, meta)
                    _collect_membership(sig, meta)
                    done += 1
                    if log_every > 0 and (done % log_every) == 0:
                        dt = time.time() - t0
                        speed = done / max(dt, 1e-6)
                        logger.info(f"Progress: {done}/{total} cones written, errors={err_cnt}, speed={speed:.2f} cone/s")
        finally:
            hb.stop()
            idx_w.close()
            _flush_membership()

        if annotate_outpin_chunks and (outpin_chunk_dir is not None) and (cone_membership_out is not None):
            annotate_outpin_chunks_with_membership(
                chunk_dir=outpin_chunk_dir,
                membership_jsonl=cone_membership_out,
                meta_key="cone_memberships",
                log_every=log_every,
            )

        logger.info(f"EP cone export finished: written={done}, errors={err_cnt}, total={total}")
        return self.out_dir

# -----------------------------
# Load utilities + DGL conversion
# -----------------------------

def load_index(index_jsonl_path: str) -> Dict[str, Dict[str, Any]]:
    table = {}
    with open(index_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            table[row["signature"]] = row
    return table

def load_chunk(signature: str, out_dir: str = chunk_dir) -> Dict[str, Any]:
    bucket = signature[:2]
    p = os.path.join(out_dir, "chunks", bucket, f"{signature}.pt")
    return torch.load(p, map_location="cpu")

def load_all_signatures(out_dir: str = chunk_dir) -> List[str]:
    """Read signatures in index.jsonl order."""
    index_path = os.path.join(out_dir, "index.jsonl")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"index.jsonl not found: {index_path}")

    sigs: List[str] = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sigs.append(row["signature"])
    return sigs

def find_signatures_by_center(center_name: str, *, k_hop: Optional[int] = None) -> List[str]:
    index_path = os.path.join(chunk_dir, "index.jsonl")
    table = load_index(index_path)

    hits = []
    for sig, row in table.items():
        meta = row.get("meta", {})
        if meta.get("center_name") != center_name:
            continue
        if k_hop is not None and int(meta.get("k_hop", -1)) != int(k_hop):
            continue
        hits.append(sig)
    return hits


def load_chunk_by_center(center_name: str, *, k_hop: Optional[int] = None) -> Dict[str, Any]:
    hits = find_signatures_by_center(center_name, k_hop=k_hop)
    if not hits:
        raise KeyError(f"No chunk found for center={center_name} (k_hop={k_hop}).")
    if len(hits) > 1:
        # multiple chunks found for center=center_name. Using the first one: hits[0]
        logger.warning(f"Multiple chunks found for center={center_name}. Using the first one: {hits[0]}")
    return load_chunk(hits[0])


def _make_bidirectional_edges_no_selfloop_dup(
    src: torch.Tensor,
    dst: torch.Tensor,
    edge_feat: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Build bidirectional edges by adding reversed edges, but DO NOT duplicate self-loops.

    Input:
      - directed edges (src[i] -> dst[i]) with aligned edge_feat (E, ...)

    Output:
      - edges: original + reversed(non-self-loop)
      - features duplicated only for reversed(non-self-loop)
    """
    if src.numel() == 0:
        return src, dst, edge_feat

    # self-loop mask
    nonself = (src != dst)
    if nonself.all():
        # fast path: no self-loops at all
        src2 = torch.cat([src, dst], dim=0)
        dst2 = torch.cat([dst, src], dim=0)
        edge_feat2 = {k: torch.cat([v, v], dim=0) if v.numel() > 0 else v for k, v in edge_feat.items()}
        return src2, dst2, edge_feat2

    # only add reverse for non-self edges
    src_rev = dst[nonself]
    dst_rev = src[nonself]

    src2 = torch.cat([src, src_rev], dim=0)
    dst2 = torch.cat([dst, dst_rev], dim=0)

    edge_feat2: Dict[str, torch.Tensor] = {}
    for k, v in edge_feat.items():
        if v.numel() == 0:
            edge_feat2[k] = v
            continue
        # v: (E, ...) -> append v[nonself] for reversed edges
        edge_feat2[k] = torch.cat([v, v[nonself]], dim=0)
    return src2, dst2, edge_feat2

def inspect_chunk_nodes(
    payload: Dict[str, Any],
    *,
    print_out: bool = True,
    max_nodes: Optional[int] = None,
):
    """
    Inspect node names stored in a chunk.

    Args:
        payload: loaded chunk payload (from load_chunk)
        print_out: whether to print
        max_nodes: optionally limit printed nodes

    Returns:
        List[str] : original node names
    """

    # Case 1 — best path (node_names saved)
    if "node_names" in payload:
        names = payload["node_names"]

    # Case 2 — fallback using global TimingGraph
    else:
        raise RuntimeError(
            "node_names not found in payload.\n"
            "Re-export chunks with `write_node_names=True` for debugging.\n"
            "Example:\n"
            "exporter.export(write_node_names=True)"
        )

    if max_nodes is not None:
        names = names[:max_nodes]

    if print_out:
        logger.info(f"\nChunk contains {len(names)} nodes:\n")
        for i, n in enumerate(names):
            logger.info(f"[{i}] {n}")

    return names

def chunk_to_dgl(
    payload: Dict[str, Any],
    *,
    hetero: bool = True,
    build_undirected_hops: bool = False,
    device: Optional[Any] = None,
    normalize: bool = False,
    norm_stats: Optional[Dict[str, Any]] = None,
    norm_fields_node: Optional[List[str]] = None,
    norm_fields_edge: Optional[List[str]] = None,
):
    """
    Convert chunk payload to DGL graph.

    Args:
        hetero: if True, split edges into (pin, cell_arc, pin) and (pin, net_arc, pin).
        build_undirected_hops: if True, create bidirectional edges in the returned DGL graph
                               by adding reversed edges (excluding self-loop duplicates) and
                               duplicating edge features accordingly.
                               (This DOES NOT change stored chunk on disk.)
    """
    try:
        import dgl
        import torch
    except Exception as e:
        raise ImportError("dgl and torch are required for chunk_to_dgl().") from e
    
    # optional normalization
    if normalize:
        if norm_stats is None:
            raise ValueError(
                "normalize=True requires passing norm_stats (preloaded) to avoid per-sample disk IO.\n"
                "Example:\n"
                "  ns = compile_norm_stats(load_norm_stats())\n"
                "  g = chunk_to_dgl(payload, normalize=True, norm_stats=ns)"
            )
        apply_norm_inplace(
            payload,
            norm_stats,
            fields_node=norm_fields_node,
            fields_edge=norm_fields_edge,
        )

    src = payload["edge_src"].to(torch.int64)
    dst = payload["edge_dst"].to(torch.int64)
    node_feat = payload["node_feat"]
    edge_feat = payload["edge_feat"]

    if not hetero:
        if build_undirected_hops:
            src_u, dst_u, ef_u = _make_bidirectional_edges_no_selfloop_dup(src, dst, edge_feat)
        else:
            src_u, dst_u, ef_u = src, dst, edge_feat

        g = dgl.graph((src_u, dst_u), num_nodes=payload["node_ids"].numel())
        for k, v in ef_u.items():
            g.edata[k] = v
        for k, v in node_feat.items():
            g.ndata[k] = v
        if device is not None:
            g = g.to(device)
        return g

    # heterograph
    is_cell = edge_feat["is_cell"].to(torch.int32)
    cell_mask = (is_cell == 1)
    net_mask = (is_cell == 0)

    src_cell = src[cell_mask]
    dst_cell = dst[cell_mask]
    ef_cell = {
        "delay": edge_feat["delay"][cell_mask],
        "sense_unate": edge_feat["sense_unate"][cell_mask],
    }

    src_net = src[net_mask]
    dst_net = dst[net_mask]
    ef_net = {
        "delay": edge_feat["delay"][net_mask],
        "sense_unate": edge_feat["sense_unate"][net_mask],
    }

    if build_undirected_hops:
        src_cell, dst_cell, ef_cell = _make_bidirectional_edges_no_selfloop_dup(src_cell, dst_cell, ef_cell)
        src_net, dst_net, ef_net = _make_bidirectional_edges_no_selfloop_dup(src_net, dst_net, ef_net)

    hg = dgl.heterograph(
        {
            ("pin", "cell_arc", "pin"): (src_cell, dst_cell),
            ("pin", "net_arc", "pin"): (src_net, dst_net),
        },
        num_nodes_dict={"pin": payload["node_ids"].numel()},
    )

    if hg.num_edges("cell_arc") > 0:
        hg.edges["cell_arc"].data["delay"] = ef_cell["delay"]
        hg.edges["cell_arc"].data["sense_unate"] = ef_cell["sense_unate"]
    if hg.num_edges("net_arc") > 0:
        hg.edges["net_arc"].data["delay"] = ef_net["delay"]
        hg.edges["net_arc"].data["sense_unate"] = ef_net["sense_unate"]

    for k, v in node_feat.items():
        hg.nodes["pin"].data[k] = v

    if device is not None:
        hg = hg.to(device)
    return hg


# -----------------------------
# Normalization stats (z-score)
# -----------------------------

def _update_running_stats(
    st: Dict[str, Dict[str, torch.Tensor]],
    key: str,
    x: torch.Tensor,
):
    """
    Maintain running stats for z-score:
      count, sum, sumsq  (over all elements)
    """
    x = x.detach().to(torch.float32).reshape(-1)
    if x.numel() == 0:
        return
    if key not in st:
        st[key] = {
            "count": torch.zeros((), dtype=torch.float64),
            "sum": torch.zeros((), dtype=torch.float64),
            "sumsq": torch.zeros((), dtype=torch.float64),
        }
    st[key]["count"] += torch.tensor(x.numel(), dtype=torch.float64)
    st[key]["sum"] += x.to(torch.float64).sum()
    st[key]["sumsq"] += (x.to(torch.float64) ** 2).sum()


def compute_and_save_norm_stats(
    *,
    out_dir: str = chunk_dir,
    signatures: Optional[List[str]] = None,
    fields_node: Optional[List[str]] = None,
    fields_edge: Optional[List[str]] = None,
    eps: float = 1e-8,
    designs: Optional[List[str]] = None,
) -> str:
    """
    Scan chunks and compute global z-score stats for selected continuous fields.
    Save to: <out_dir>/norm_stats.pt

    NOTE:
      - Should be computed on TRAIN split only to avoid leakage.
      - Here we compute over provided signatures; if None, use all signatures in index.jsonl.
      - When *designs* is given it is stored in ``meta["designs"]`` for cache validation.
    """
    if fields_node is None:
        fields_node = ["slack", "arrival", "trans", "ceff", "bbox", "level"]
    if fields_edge is None:
        fields_edge = ["delay"]

    if signatures is None:
        signatures = load_all_signatures(out_dir=out_dir)

    stats: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {"node": {}, "edge": {}}

    for sig in tqdm.tqdm(signatures, desc="Computing norm_stats", colour="cyan"):
        payload = load_chunk(sig, out_dir=out_dir)  # cpu
        nf = payload.get("node_feat", {})
        ef = payload.get("edge_feat", {})

        # node fields
        for k in fields_node:
            if k in nf:
                _update_running_stats(stats["node"], k, nf[k])

        # edge fields
        for k in fields_edge:
            if k in ef:
                _update_running_stats(stats["edge"], k, ef[k])

    # finalize mean/std
    norm_stats = {"node": {}, "edge": {}, "meta": {}}
    for scope in ["node", "edge"]:
        for k, st in stats[scope].items():
            cnt = st["count"].clamp_min(1.0)
            mean = st["sum"] / cnt
            var = (st["sumsq"] / cnt) - mean**2
            var = torch.clamp(var, min=0.0)
            std = torch.sqrt(var + eps)

            norm_stats[scope][k] = {
                "mean": mean.to(torch.float32),
                "std": std.to(torch.float32),
            }

    norm_stats["meta"] = {
        "fields_node": fields_node,
        "fields_edge": fields_edge,
        "eps": float(eps),
        "num_signatures": int(len(signatures)),
        "designs": sorted(designs) if designs is not None else None,
    }

    p = os.path.join(out_dir, "norm_stats.pt")
    torch.save(norm_stats, p)
    logger.info(f"Saved norm_stats to: {p}")
    return p


def load_norm_stats(*, out_dir: str = chunk_dir) -> Dict[str, Any]:
    p = os.path.join(out_dir, "norm_stats.pt")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"norm_stats.pt not found: {p}")
    return torch.load(p, map_location="cpu")

def compile_norm_stats(
    norm_stats: Dict[str, Any],
    *,
    fields_node: Optional[List[str]] = None,
    fields_edge: Optional[List[str]] = None,
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """
    Precompute mean + inv_std for fast normalization.
    Also filter to only specified fields (whitelist).
    """
    out = {"node": {}, "edge": {}, "meta": dict(norm_stats.get("meta", {}))}
    if fields_node is None:
        fields_node = list(norm_stats.get("node", {}).keys())
    if fields_edge is None:
        fields_edge = list(norm_stats.get("edge", {}).keys())

    for k in fields_node:
        ms = norm_stats.get("node", {}).get(k, None)
        if ms is None:
            continue
        mean = ms["mean"].to(torch.float32)
        std = ms["std"].to(torch.float32)
        inv_std = 1.0 / (std + eps)
        out["node"][k] = {"mean": mean, "inv_std": inv_std}

    for k in fields_edge:
        ms = norm_stats.get("edge", {}).get(k, None)
        if ms is None:
            continue
        mean = ms["mean"].to(torch.float32)
        std = ms["std"].to(torch.float32)
        inv_std = 1.0 / (std + eps)
        out["edge"][k] = {"mean": mean, "inv_std": inv_std}

    out["meta"]["eps"] = float(eps)
    out["meta"]["fields_node_used"] = list(out["node"].keys())
    out["meta"]["fields_edge_used"] = list(out["edge"].keys())
    return out

def apply_norm_inplace(
    payload: Dict[str, Any],
    norm_stats: Dict[str, Any],
    *,
    fields_node: Optional[List[str]] = None,
    fields_edge: Optional[List[str]] = None,
    mark_key: str = "_normalized",
) -> Dict[str, Any]:
    """
    Fast in-place z-score:
      x := (x - mean) * inv_std

    - supports field whitelist via fields_node/fields_edge
    - skips if payload already normalized (mark_key)
    - expects norm_stats to be either:
        {scope:{k:{mean,std}}}  OR  {scope:{k:{mean,inv_std}}}
      (compile_norm_stats produces inv_std)
    """
    if payload.get(mark_key, False):
        return payload

    nf = payload.get("node_feat", {})
    ef = payload.get("edge_feat", {})

    node_stats = norm_stats.get("node", {})
    edge_stats = norm_stats.get("edge", {})

    if fields_node is None:
        fields_node = list(node_stats.keys())
    if fields_edge is None:
        fields_edge = list(edge_stats.keys())

    # node
    for k in fields_node:
        x = nf.get(k, None)
        if x is None:
            continue
        ms = node_stats.get(k, None)
        if ms is None:
            continue

        mean = ms["mean"]
        inv_std = ms.get("inv_std", None)
        if inv_std is None:
            std = ms["std"]
            inv_std = 1.0 / (std + 1e-8)

        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        nf[k] = (x - mean) * inv_std

    # edge
    for k in fields_edge:
        x = ef.get(k, None)
        if x is None:
            continue
        ms = edge_stats.get(k, None)
        if ms is None:
            continue

        mean = ms["mean"]
        inv_std = ms.get("inv_std", None)
        if inv_std is None:
            std = ms["std"]
            inv_std = 1.0 / (std + 1e-8)

        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        ef[k] = (x - mean) * inv_std

    payload["node_feat"] = nf
    payload["edge_feat"] = ef
    payload[mark_key] = True
    return payload

def build_timing_graph(
    design: str,
    **tg_kwargs: Any,
) -> Any:
    """Build a TimingGraph and remove unpropagated arcs.

    Returns the ready-to-use TimingGraph object.  Call this once per
    design and pass the result to
    ``build_timing_graph_and_export`` / ``build_timing_graph_and_export_ep_cones``
    to avoid redundant construction.
    """
    logger.info(f"Build TimingGraph: design={design}")
    TG = TimingGraph(design)

    if hasattr(TG, "remove_unpropagated_arcs"):
        TG.remove_unpropagated_arcs()
    return TG


def build_packed_timing_graph(
    design: str,
    *,
    TG: Optional[Any] = None,
    **tg_kwargs: Any,
) -> Tuple[Any, "PackedTimingGraph"]:
    """Build (or reuse) a TimingGraph **and** its PackedTimingGraph.

    Returns ``(TG, packed)`` so callers can hand both to the export helpers.
    If *TG* is already provided it is reused; otherwise one is built via
    :func:`build_timing_graph`.
    """
    if TG is None:
        TG = build_timing_graph(design, **tg_kwargs)
    design_id = design
    packed = PackedTimingGraph(TG.G, design_id=design_id).pack()
    return TG, packed


def build_timing_graph_and_export(
    design: str,
    *,
    out_dir: str = chunk_dir,
    neg_only: bool = True,
    tg_kwargs: Optional[Dict[str, Any]] = None,
    export_kwargs: Optional[Dict[str, Any]] = None,
    TG: Optional[Any] = None,
    packed: Optional["PackedTimingGraph"] = None,
) -> str:
    """
    Given a design, build TimingGraph and export outpin-centered chunks to disk.

    Args:
        design: design name, e.g. "aes_cipher_top"
        out_dir: root output directory (default chunk_dir)
        tg_kwargs: forwarded to TimingGraph(...) constructor (if you later add more args)
        export_kwargs: forwarded to exporter.export(...)
        TG: pre-built TimingGraph (skip construction if provided)
        packed: pre-built PackedTimingGraph (skip packing if provided)

    Returns:
        str: the actual export directory that contains index.jsonl and chunks/
    """
    tg_kwargs = tg_kwargs or {}
    export_kwargs = export_kwargs or {}

    design_id = design
    out_dir_p = Path(out_dir)

    if TG is None:
        TG = build_timing_graph(design, **tg_kwargs)

    logger.info(f"Export chunks to: {str(out_dir_p)}")
    exporter = OutpinChunkExporter(TG, out_dir=str(out_dir_p), design_id=design_id, packed=packed)
    
    if neg_only:
        ep_names = TG.get_setup_neg_ep()
    else:
        ep_names = None

    # sensible defaults (you can override in export_kwargs)
    defaults = dict(
        k_hop=3,
        exclude_clock_network=False,
        exclude_async_pin=False,
        fanout_cap=None,
        num_workers=min(16, mp.cpu_count() - 1),
        write_node_names=True,
        outpin_filter=None,
        ep_names=ep_names,
        chunksize=64,
        save_norm_stats=False,
        norm_stats_signatures=None,
        mp_start_method="forkserver",
        log_every=2000,
    )
    # user overrides win
    defaults.update(export_kwargs)

    return exporter.export(**defaults)


def build_timing_graph_and_export_ep_cones(
    design: str,
    *,
    out_dir: str = ep_dir,
    TG: Optional[Any] = None,
    packed: Optional["PackedTimingGraph"] = None,
    export_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Given a design, build TimingGraph and export EP cones to disk.

    Args:
        design: design name, e.g. "aes_cipher_top"
        out_dir: root output directory (default ep_dir)
        TG: pre-built TimingGraph (skip construction if provided)
        packed: pre-built PackedTimingGraph (skip packing if provided)
        export_kwargs: forwarded to exporter.export(...)

    Returns:
        str: the actual export directory that contains index.jsonl and cones/
    """
    if TG is None:
        TG = build_timing_graph(design)

    export_kwargs = export_kwargs or {}
    defaults = dict(
        endpoints=TG.get_setup_neg_ep(),
        cone_dir="fanin",
        max_hops=None,
        max_nodes=100000,
        num_workers=min(8, mp.cpu_count() - 1),
        write_node_names=True,
        cone_membership_out=None,
    )
    defaults.update(export_kwargs)

    exporter = EndpointLogicConeExporter(
        TG, out_dir=out_dir, design_id=design, packed=packed,
    )
    return exporter.export(**defaults)

def debug_compare_chunk_vs_raw(TG, payload: dict, k: Optional[int] = None, *, strict: bool = True):
    """
    [DEBUG ONLY] Compare chunk nodes vs raw nodes
    Compare:
      - raw undirected k-hop neighborhood from TG.G
      - nodes stored in payload["node_names"]

    Key fix:
      - center_name MUST come from payload["storage_key"]["center_name"]
      - optionally check user-provided center mismatch (strict)
    """
    G = TG.G

    payload_center = payload["storage_key"]["center_name"]
    if k is None:
        k = int(payload["meta"]["k_hop"])

    # chunk nodes
    chunk_nodes = payload.get("node_names", None)
    if chunk_nodes is None:
        raise RuntimeError("payload has no node_names; re-export with write_node_names=True")

    # sanity: chunk must contain its own center
    if payload_center not in set(chunk_nodes):
        raise RuntimeError(
            f"Chunk node_names does not contain payload center!\n"
            f"payload_center={payload_center}\n"
            f"maybe node_names generation is broken."
        )

    # raw: undirected k-hop (no filter)
    visited = set([payload_center])
    frontier = [payload_center]
    for _ in range(k):
        nxt = set()
        for u in frontier:
            for v in G.predecessors(u):
                if v not in visited:
                    nxt.add(v)
            for v in G.successors(u):
                if v not in visited:
                    nxt.add(v)
        visited |= nxt
        frontier = list(nxt)
        if not frontier:
            break

    raw_nodes = sorted(visited)
    chunk_set = set(chunk_nodes)

    only_raw = [n for n in raw_nodes if n not in chunk_set]
    only_chunk = [n for n in chunk_nodes if n not in set(raw_nodes)]

    print(f"Center (from payload): {payload_center}")
    print(f"Raw undirected {k}-hop nodes = {len(raw_nodes)}")
    print(f"Chunk nodes                 = {len(chunk_nodes)}")

    print("\nIn raw but NOT in chunk (first 50):")
    for n in only_raw[:50]:
        print(" ", n, "is_clock_network=", G.nodes[n].get("is_clock_network", None))

    print("\nIn chunk but NOT in raw (first 50):")
    for n in only_chunk[:50]:
        print(" ", n)