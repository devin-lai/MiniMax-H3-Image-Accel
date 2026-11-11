# Contributing

For a bug report, include the GPU model and count, driver version, server profile, exact command, and the relevant error. For performance changes, use a matched baseline and record the settings described in the [benchmarking guide](docs/benchmarking.md). Keep access tokens, private paths, and prompts out of logs you share. Review the [model agreements](THIRD_PARTY_NOTICES.md#model-files-and-results) before sharing generated material. [Open an issue](https://github.com/devin-lai/MiniMax-H3-Image-Accel/issues/new)

## CPU checks

From the repo root, run the same tests as GitHub Actions in a Python 3.12 environment.

```bash
python -m pip install numpy==2.3.5 pillow==12.3.0
PYTHONPATH=tests python -m unittest \
  test_launch_profile test_compare_run_stack test_export_accuracy test_validate_accuracy
```

The tests cover launch policy and comparison tooling with synthetic fixtures. They run without a checkpoint or CUDA.

## CUDA checks

On the supported Linux runtime, run the existing GPU checks after `bin/install.sh`.

```bash
H3_TEST_GPU=0 bin/test.sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tests/test_exact_vae_graph.py
```

The exact VAE replay tests exercise CUDA graphs on SM89. The quantized Ulysses tests also need SM89 and SageAttention2. For changes to BF16 transport, run [`tests/test_lossless_collectives.py`](tests/test_lossless_collectives.py) with four visible CUDA GPUs and the vendored SGLang runtime on `PYTHONPATH`. Record any skipped device tests with the hardware used.
