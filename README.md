# MARQ Qwen3 benchmark branch

This isolated branch contains the reproducible MARQ v0.2.0 source archive and a GitHub Actions smoke run for `Qwen/Qwen3-0.6B` pinned to revision `c1899de289a04d12100db370d81485cdf75e47ca`.

The workflow unpacks the archive, runs the full unit-test suite, detects the available CPU/CUDA backend, downloads the pinned model, performs a one-block end-to-end MARQ calibration smoke test, and uploads the result manifest and packed export as a workflow artifact.

The main branch is intentionally untouched.
