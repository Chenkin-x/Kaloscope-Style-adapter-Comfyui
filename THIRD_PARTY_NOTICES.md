# Sources and maintenance boundary

- Artist plugin interface and original implementation: [spawner1145/comfyui-kaloscope](https://github.com/spawner1145/comfyui-kaloscope), original integration baseline `1f1f12d4103fd10c6f890ecf197a0e280d1979d5`. GPL-3.0, retained in `LICENSE`. This generation package is maintained separately; it is not a replacement for the artist-analysis plugin.
- DINOv3 ViT inference source: [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3), copied from the original integration's vendored DINOv3. The selected `layers/`, `models/vision_transformer.py`, and tensor helpers live under `kaloscope_style/_vendor/dinov3/`. Original copyright headers and `LICENSE.md` are preserved. Imports were made package-relative; model operations were not changed. No hub download code or unrelated backbones are included.
- `encoder.py` is the generation-specific loader. It shares the original folder/config/checkpoint convention and 256D projector contract, while omitting classifier construction and LSNet. It is intentionally not a general DINO feature-analysis implementation.
- Krea2/MOSS integration modules, checkpoint export, artist prototype lookup, frontend search and example workflows remain in this package. Official ComfyUI and MOSS modules are runtime dependencies; they are not vendored here.
- Upstream code licenses do not replace model weight licenses or grant rights to external datasets. See the accompanying model release documentation for weight provenance and terms.

Maintain the shared `KALOSCOPE_MODEL` input contract, reference preprocessing, and numerical parity when updating the encoder. Keep runtime imports relative; do not add global `sys.path` or module aliases. Regression tests belong in `tests/`; full-weight parity evidence belongs in `docs/validation/`.

Legacy checkpoint signatures, filename aliases and workflow identifiers are isolated in `kaloscope_style/compat.py`. They remain solely for reading existing assets; public nodes, sockets, examples and user documentation use functional names. Model tensor data and numerical inference operations are unchanged by this naming update.
