"""
Resolve every training run to the exact data files it consumed.

The mapping is not guesswork and does not depend on the checkpoints still
existing on disk: each training script carries a `_DATASET_LOOKUP` table
keyed by `(model_size, chinchilla_multiplier)`, and for the repetition
stream that table's values resolve through `DataMix` to a file list that
also lives in this repo.  So the chain

    (size, chinchilla_scale)  ->  _DATASET_LOOKUP  ->  DataMix  ->  *.txt
                                                    ->  train_<X>B.npy

is fully reconstructible from source, which matters because netscratch's
90-day purge has already removed most of the `data_paths.txt` records the
runs themselves wrote.

Tables read (parsed, never imported — importing pulls in torch):
  src/scripts/official/OLMo-scale-train-multiepoch-dolma.py   repeat
  src/scripts/official/OLMo-scale-train-paraphrase-dolma.py   paraphrase
  src/scripts/official/OLMo-scale-train-selfdistill-dolma.py  selfdistill
  src/olmo_core/data/mixes/__init__.py                        DataMix values
  src/olmo_core/data/mixes/syn_data_scaling/dolma/*.txt       file lists

Usage:
    python run_data_provenance.py --verify   # check against surviving checkpoints
    python run_data_provenance.py --dump     # print the resolved table
"""

import argparse
import ast
import glob
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

TRAIN_SCRIPTS = {
    "repeat": "src/scripts/official/OLMo-scale-train-multiepoch-dolma.py",
    "paraphrase": "src/scripts/official/OLMo-scale-train-paraphrase-dolma.py",
    "selfdistill": "src/scripts/official/OLMo-scale-train-selfdistill-dolma.py",
}
MIXES_INIT = "src/olmo_core/data/mixes/__init__.py"
MIX_TXT_DIR = "src/olmo_core/data/mixes/syn_data_scaling/dolma"

DOLMA_SUBPATH = "preprocessed/dolma2-0625/resharded"
PARAPHRASE_SUBPATH = "paraphrased/sized_smollm2_mixed"
PARAPHRASE_CORPUS = "sized_smollm2_mixed"          # SmolLM2-1.7B-Instruct, 4 mixed prompts

CHECKPOINT_ROOT = "/n/netscratch/barak_lab/Lab/sqin/olmo/checkpoints"


# ---------------------------------------------------------------------------
# Parsing the in-repo lookup tables
# ---------------------------------------------------------------------------

def _dataset_lookup(path):
    """Extract `_DATASET_LOOKUP` as {(model_size, chin_float): value_str}.

    Keys are literal tuples; values are either a string literal (paraphrase,
    selfdistill) or a `DataMix.NAME` attribute (repeat), so the dict cannot
    go through ast.literal_eval wholesale.
    """
    tree = ast.parse(open(os.path.join(REPO_ROOT, path)).read())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(t, "id", None) == "_DATASET_LOOKUP" for t in node.targets):
            continue
        out = {}
        for k, v in zip(node.value.keys, node.value.values):
            model, chin = ast.literal_eval(k)
            if isinstance(v, ast.Attribute):          # DataMix.OLMo_dolma_7_4B
                out[(model, float(chin))] = ("datamix", v.attr)
            else:                                     # "train_2.24B"
                out[(model, float(chin))] = ("shard", ast.literal_eval(v))
        return out
    raise RuntimeError(f"_DATASET_LOOKUP not found in {path}")


def _datamix_values():
    """{enum_attr_name: enum_value} for the OLMo_dolma_* members."""
    src = open(os.path.join(REPO_ROOT, MIXES_INIT)).read()
    return dict(re.findall(r'^\s+(OLMo_dolma_\w+)\s*=\s*"([^"]+)"', src, re.M))


def _mix_basenames(mix_value):
    """Filenames listed by a DataMix's .txt manifest, in order."""
    txt = os.path.join(REPO_ROOT, MIX_TXT_DIR, f"{mix_value}.txt")
    names = []
    for line in open(txt):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.append(os.path.basename(line.split(",", 1)[-1]))
    return names


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

class Resolver:
    def __init__(self):
        self.lookup = {s: _dataset_lookup(p) for s, p in TRAIN_SCRIPTS.items()}
        self.mixvals = _datamix_values()
        self._mixcache = {}

    def _files_for(self, kind, value):
        if kind == "shard":
            return [f"{value}.npy"], value
        mix_value = self.mixvals[value]
        if mix_value not in self._mixcache:
            self._mixcache[mix_value] = _mix_basenames(mix_value)
        return self._mixcache[mix_value], mix_value

    def resolve(self, stream, size, scale, K=None):
        """-> dict of provenance columns, or None if the cell is not in the table."""
        model = size.upper()                          # '370m' -> '370M'
        entry = self.lookup[stream].get((model, float(scale)))
        if entry is None:
            return None
        kind, value = entry
        d_files, mix = self._files_for(kind, value)

        if stream == "paraphrase":
            shard = value                             # 'train_2.24B'
            k = int(K)
            dprime = [f"{shard}_seed{i}.npy" for i in range(1, k + 1)]
            return dict(
                D_files=";".join(d_files),
                D_dir=DOLMA_SUBPATH,
                D_prime_files=";".join(dprime),
                D_prime_dir=f"{DOLMA_SUBPATH}/{PARAPHRASE_SUBPATH}",
                D_prime_corpus=PARAPHRASE_CORPUS,
                paraphrase_seeds=f"1-{k}" if k > 1 else "1",
                data_mix="",
            )

        if stream == "selfdistill":
            return dict(
                D_files=";".join(d_files),
                D_dir=DOLMA_SUBPATH,
                D_prime_files="",                     # teacher-generated; see manifest
                D_prime_dir=f"{DOLMA_SUBPATH}/self_distill",
                D_prime_corpus="teacher_manifest.json",
                paraphrase_seeds="",
                data_mix="",
            )

        return dict(                                  # repeat: D only, re-read `epochs` times
            D_files=";".join(d_files),
            D_dir=DOLMA_SUBPATH,
            D_prime_files="",
            D_prime_dir="",
            D_prime_corpus="",
            paraphrase_seeds="",
            data_mix=mix,
        )


PROVENANCE_FIELDS = ["D_files", "D_dir", "D_prime_files", "D_prime_dir",
                     "D_prime_corpus", "paraphrase_seeds", "data_mix"]


# ---------------------------------------------------------------------------
# Verification against surviving checkpoints
# ---------------------------------------------------------------------------

def _norm_scale(scale):
    f = float(scale)
    return str(int(f)) if f == int(f) else str(f)


def ground_truth(stream_scale_run):
    """Read a run's own data_paths.txt, if the purge has not eaten it."""
    scale, run_name = stream_scale_run
    d = os.path.join(CHECKPOINT_ROOT, f"chinchilla_{_norm_scale(scale)}", run_name)
    steps = sorted(glob.glob(os.path.join(d, "step*", "data_paths.txt")))
    if not steps:
        return None
    with open(steps[-1]) as f:
        return [os.path.basename(l.strip()) for l in f if l.strip()]


def verify(csv_path):
    import csv as _csv
    res = Resolver()
    rows = list(_csv.DictReader(open(csv_path)))
    ok = bad = nogt = unresolved = 0
    problems = []
    for r in rows:
        p = res.resolve(r["stream"], r["size"], r["chinchilla_scale"],
                        r["K"] or None)
        if p is None:
            unresolved += 1
            problems.append(("UNRESOLVED", r["stream"], r["size"],
                             r["chinchilla_scale"], r["K"]))
            continue
        gt = ground_truth((r["chinchilla_scale"], r["run_name"]))
        if gt is None:
            nogt += 1
            continue
        expect = [f for f in p["D_files"].split(";") if f]
        if p["D_prime_files"]:
            expect += p["D_prime_files"].split(";")
        if r["stream"] == "selfdistill":
            # The synthetic file list is produced by globbing the teacher's
            # output dir, so it is not in the lookup table; check D only.
            got, exp = gt[:1], expect[:1]
        else:
            got, exp = gt, expect
        if got == exp:
            ok += 1
        else:
            bad += 1
            problems.append(("MISMATCH", r["stream"], r["size"],
                             r["chinchilla_scale"], r["K"],
                             f"expected={exp[:2]}", f"got={got[:2]}"))
    print(f"verified against surviving checkpoints: {ok} match, {bad} mismatch, "
          f"{nogt} no on-disk record (purged), {unresolved} not in lookup table")
    seen = set()
    for p in problems:
        if p[:5] in seen:
            continue
        seen.add(p[:5])
        print("   ", *p)
    return bad == 0 and unresolved == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--csv", default=os.path.join(
        SCRIPT_DIR, "data_export/hparam_sweeps/hparam_sweep_all.csv"))
    args = ap.parse_args()

    if args.dump:
        res = Resolver()
        for stream in TRAIN_SCRIPTS:
            for (model, chin), _ in sorted(res.lookup[stream].items()):
                p = res.resolve(stream, model.lower(), chin, K=1)
                print(f"{stream:12s} {model:5s} chin={chin:<6g} "
                      f"D={p['D_files']}"
                      + (f"  D'={p['D_prime_files']}" if p["D_prime_files"] else ""))
    if args.verify:
        sys.exit(0 if verify(args.csv) else 1)


if __name__ == "__main__":
    main()
