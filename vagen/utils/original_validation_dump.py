"""Opt-in, observation-only persistence for original single-row validation."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def dump_validation_batch(output_dir, env_configs, recordings) -> Path:
    """Persist exactly one input/record pair; never infer async multi-row order.

    An exclusively reserved row directory contains PNG payloads followed by the
    fsynced record.json readiness marker. Interrupted rows remain inspectable and
    cannot be overwritten or accepted as complete.
    """
    if len(env_configs) != 1 or len(recordings) != 1:
        raise ValueError('original validation dump requires exactly one input and one recording')
    info, record = env_configs[0], recordings[0]
    identity = {key: info[key] for key in ('source_index', 'source_key', 'seed')}
    identity['eval_set'] = info['env_config']['eval_set']
    # numpy integer scalars can originate from the original parquet loader.
    for key in ('source_index', 'seed'):
        value = identity[key]
        if type(value).__module__.startswith('numpy') and hasattr(value, 'item'):
            value = value.item()
        if type(value) is not int or value < 0:
            raise ValueError(f'invalid {key}')
        identity[key] = value
    if (identity['eval_set'] not in ('base', 'common_sense')
            or identity['source_key'] != f"{identity['eval_set']}:{identity['seed']}"):
        raise ValueError('invalid source identity')
    if type(record['metrics'].get('success')) is not bool:
        raise ValueError('metrics.success must be an exact boolean')
    if not isinstance(record.get('output_str'), str) or not record.get('history'):
        raise ValueError('missing response trajectory/history')
    if not record.get('image_data'):
        raise ValueError('missing rollout images')
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _sync_dir(root.parent)
    for existing in root.glob('row_*/record.json'):
        saved = json.loads(existing.read_text())
        if (saved['source_key'] == identity['source_key']
                and saved['source_index'] != identity['source_index']):
            raise ValueError('duplicate source_key under a different source_index')
    row_dir = root / f"row_{identity['source_index']:06d}"
    row_dir.mkdir(exist_ok=False)
    _sync_dir(root)
    image_refs = {}

    def encode(value):
        # Pillow is needed only when the opt-in writer is actually invoked.
        from PIL import Image
        if isinstance(value, Image.Image):
            if id(value) not in image_refs:
                name = f'image_{len(image_refs):04d}.png'
                path = row_dir / name
                with path.open('xb') as stream:
                    value.copy().save(stream, format='PNG')
                    stream.flush()
                    os.fsync(stream.fileno())
                image_refs[id(value)] = {
                    'path': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'mode': value.mode, 'size': list(value.size),
                }
            return {'image_file': image_refs[id(value)]}
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise ValueError('non-string audit key')
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float and math.isfinite(value):
            return value
        if type(value).__module__.startswith('numpy') and getattr(value, 'ndim', None) == 0:
            return encode(value.item())
        raise ValueError(f'unsupported audit value: {type(value).__name__}')

    payload = {'format': 'original_validation_row_v1', **identity,
               'env_config': encode(info), 'recording': encode(record)}
    serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    _sync_dir(row_dir)
    with (row_dir / 'record.json').open('x') as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_dir(row_dir)
    return row_dir / 'record.json'
