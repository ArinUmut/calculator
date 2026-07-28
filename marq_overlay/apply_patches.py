from pathlib import Path

path = Path('marq-qwen/marq/quantizer.py')
source = path.read_text()

old_bounds = 'return -(1 << (bits - 1)), (1 << (bits - 1)) - 1'
new_bounds = 'limit = (1 << (bits - 1)) - 1\n    return -limit, limit'
if old_bounds not in source:
    raise RuntimeError('expected asymmetric signed_bounds implementation not found')
source = source.replace(old_bounds, new_bounds, 1)

old_hardening = '''    deq2 = scales2[groups] * hard_codes.float()\n    e = deq2 - anchor\n    loss = float((linear * e).sum() + 0.5 * op_pred.quadratic(e, include_damping=False))\n    return QuantizedCandidate(\n'''
new_hardening = '''    deq2 = scales2[groups] * hard_codes.float()\n    e = deq2 - anchor\n    loss = float((linear * e).sum() + 0.5 * op_pred.quadratic(e, include_damping=False))\n    bootstrap_e = deq - anchor\n    bootstrap_loss = float(\n        (linear * bootstrap_e).sum()\n        + 0.5 * op_pred.quadratic(bootstrap_e, include_damping=False)\n    )\n    if loss > bootstrap_loss:\n        hard_codes = codes\n        scales2 = scales\n        deq2 = deq\n        loss = bootstrap_loss\n    return QuantizedCandidate(\n'''
if old_hardening not in source:
    raise RuntimeError('expected hard candidate block not found')
source = source.replace(old_hardening, new_hardening, 1)
path.write_text(source)
