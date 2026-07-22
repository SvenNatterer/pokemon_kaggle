import glob
import os
import re


MODEL_PREFIXES = ("ppo_v6_deck", "ppo_v5b_deck", "ppo_v5_deck", "ppo_belief_deck", "ppo_v4_deck", "ppo_deck")
DEFAULT_MODEL_PREFIX = "ppo_v5_deck"
MODEL_FILE_RE = re.compile(
    r"^(?P<prefix>ppo(?:_belief|_v4|_v5b?|_v6)?_deck)_(?P<deck_id>(?:bank_)?\d+)"
    r"(?P<variant>_.*)?\.zip$"
)


def default_deck_model_path(deck_id, model_dir="models"):
    return os.path.join(model_dir, f"{DEFAULT_MODEL_PREFIX}_{deck_id}.zip")


def parse_deck_model_path(path):
    match = MODEL_FILE_RE.match(os.path.basename(path))
    if not match:
        return None

    data = match.groupdict()
    data["deck_id"] = str(data["deck_id"])
    data["variant"] = data.get("variant") or ""
    data["path"] = path
    data["name"] = os.path.splitext(os.path.basename(path))[0]
    return data


def strip_zip_suffix(path):
    return path[:-4] if path.endswith(".zip") else path


def _candidate_sort_key(path):
    parsed = parse_deck_model_path(path) or {}
    variant = parsed.get("variant", "")
    prefix = parsed.get("prefix", "")

    # Variant sorting is only used by explicit legacy discovery. Runtime model
    # resolution defaults to exact final filenames and never selects a variant.
    variant_rank = 2 if not variant else 1 if variant.startswith("_checkpoint_") else 0
    prefix_rank = len(MODEL_PREFIXES) - MODEL_PREFIXES.index(prefix) if prefix in MODEL_PREFIXES else 0
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0
    return (mtime, variant_rank, prefix_rank, os.path.basename(path))


def deck_model_candidates(deck_id, model_dir="models", include_variants=True):
    deck_id = str(deck_id)
    candidates = set()

    for prefix in MODEL_PREFIXES:
        exact = os.path.join(model_dir, f"{prefix}_{deck_id}.zip")
        if os.path.exists(exact):
            candidates.add(exact)

        if include_variants:
            pattern = os.path.join(model_dir, f"{prefix}_{deck_id}_*.zip")
            for path in glob.glob(pattern):
                if parse_deck_model_path(path):
                    candidates.add(path)

    return sorted(candidates, key=_candidate_sort_key, reverse=True)


def iter_existing_deck_model_paths(deck_id, model_dir="models", include_ghost=False, include_variants=True):
    search_dirs = [model_dir]
    if include_ghost:
        search_dirs.append(os.path.join(model_dir, "ghost_pool"))

    for directory in search_dirs:
        yield from deck_model_candidates(deck_id, directory, include_variants=include_variants)


def resolve_deck_model_path(deck_id, model_dir="models", include_ghost=False, include_variants=False):
    return next(iter_existing_deck_model_paths(deck_id, model_dir, include_ghost, include_variants), "")


def resolve_deck_model_base(deck_id, model_dir="models"):
    path = resolve_deck_model_path(deck_id, model_dir, include_variants=False)
    if path:
        return path[:-4] if path.endswith(".zip") else path
    return default_deck_model_path(deck_id, model_dir)[:-4]


def discover_deck_models(model_dir="models", include_ghost=False, include_variants=True):
    search_dirs = [model_dir]
    if include_ghost:
        search_dirs.append(os.path.join(model_dir, "ghost_pool"))

    models = []
    seen = set()
    for directory in search_dirs:
        for path in glob.glob(os.path.join(directory, "*.zip")):
            parsed = parse_deck_model_path(path)
            if not parsed or (parsed["variant"] and not include_variants):
                continue
            if path in seen:
                continue
            seen.add(path)
            models.append(parsed)

    return sorted(models, key=lambda item: _candidate_sort_key(item["path"]), reverse=True)
