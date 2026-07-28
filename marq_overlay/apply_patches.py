from pathlib import Path

path = Path('marq-qwen/marq/quantizer.py')
source = path.read_text()
old = 'return -(1 << (bits - 1)), (1 << (bits - 1)) - 1'
new = 'limit = (1 << (bits - 1)) - 1\n    return -limit, limit'
if old not in source:
    raise RuntimeError('expected asymmetric signed_bounds implementation not found')
path.write_text(source.replace(old, new, 1))
