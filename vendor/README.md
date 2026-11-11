# Runtime sources

- [`sglang-h3.patch`](sglang-h3.patch) contains the inference and test changes for the SGLang revision pinned in [`sources.json`](../provenance/sources.json). The installer checks out that revision, applies the patch, and exposes the cached runtime at `vendor/sglang/`. To prepare just this source tree, run `python3 bin/prepare_sglang.py` from the repo root.
- [`SageAttention/`](SageAttention/) includes SageAttention2, the optional SageAttention3 source, and its CUTLASS build headers. See the [third-party notices](../THIRD_PARTY_NOTICES.md) for revisions and licenses.
