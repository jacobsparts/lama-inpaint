#!/usr/bin/env python3
"""Export big-lama's TorchScript generator weights to a flat FP32 blob + JSON index.

The Rust engine reads this instead of the 205 MB TorchScript zip, so no pickle
parsing and no torch dependency is needed at runtime.

Layout: all tensors concatenated in a fixed order, 64-byte aligned, little
endian FP32.  ``weights.json`` records, per tensor:

  name, shape, offset (bytes), nbytes

plus the ordered list used for the network definition.  Run once:

  python3 export_weights.py models/big-lama.pt models/big-lama.bin
"""

import argparse
import json
import os
import struct
import sys

import torch

ALIGN = 64


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('checkpoint')
    ap.add_argument('out_bin')
    ap.add_argument('--out-json', default=None)
    args = ap.parse_args()
    out_json = args.out_json or (os.path.splitext(args.out_bin)[0] + '.json')

    torch.set_num_threads(1)
    blob = torch.jit.load(args.checkpoint, map_location='cpu')
    sd = blob.generator.state_dict()
    del blob

    entries = []
    blobs = []
    off = 0

    def pad(n):
        return (-n) % ALIGN

    for name, t in sd.items():
        t = t.detach().to(torch.float32).contiguous().cpu()
        raw = t.numpy().tobytes()
        p = pad(off)
        if p:
            blobs.append(b'\x00' * p)
            off += p
        entries.append({
            'name': name,
            'shape': list(t.shape),
            'offset': off,
            'nbytes': len(raw),
        })
        blobs.append(raw)
        off += len(raw)

    with open(args.out_bin, 'wb') as fh:
        for b in blobs:
            fh.write(b)

    total = sum(e['nbytes'] for e in entries)
    with open(out_json, 'w') as fh:
        json.dump({
            'format': 'lama-fp32-v1',
            'align': ALIGN,
            'count': len(entries),
            'total_bytes': total,
            'tensors': entries,
        }, fh, indent=1)

    print(f'wrote {args.out_bin} ({off} bytes, {total} tensor bytes, '
          f'{len(entries)} tensors)')
    print(f'wrote {out_json}')


if __name__ == '__main__':
    sys.exit(main())
