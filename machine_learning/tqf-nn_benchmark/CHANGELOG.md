# TQF-NN Benchmark Tools: Changelog

**We're fired up and forging some new experimental tools---it's a work in progress!**

All notable changes to these TQF-NN benchmark tools will be documented in this file. For some potential future advancements, see [`FUTURE_TODO.md`](FUTURE_TODO.md).

---

## [1.0.1] - 2026-02-12

### Added
- Persistent result logging to JSON files in `data/output/` directory. The results can be logged to a custom directory via the `--results-dir` CLI parameter and result logging can be disabled via the `--no-save-results` CLI parameter.
- New TQF D₆ and 𝕋₂₄ orbit mixing features with CLI parameter flags `--tqf-use-d6-orbit-mixing` and `--tqf-use-t24-orbit-mixing`, respectively.
- New CLI parameter flag `--no-tqf-z6-augmentation` to optionally disable the ℤ₆ data augmentation feature (which is enabled by default). Recommended when using the TQF orbit mixing features (e.g., because the ℤ₆ data augmentation and TQF ℤ₆ orbit mixing can conflict/compete and actually negatively impact the accuracy).
- Data loading/preprocessing performance and caching optimizations to eliminate image input data loading bottleneck.
- More automated tests for symmetry operations.
- A pregame message about caching puppies---an extremely critical and fundamental addition.

### Changed
- TQF ℤ₆ orbit mixing CLI parameter flag from `--tqf-use-orbit-mixing` to `--tqf-use-z6-orbit-mixing` to distinguish between the other TQF orbit mixing features.

### Removed
- CLI fractal parameter consolidation/simplification (removed `--tqf-fractal-dim-tolerance` and `--tqf-box-counting-scales`). The user just doesn't need to be tuning these via the CLI all the time.

## [1.0.0] - 2026-02-07

### Added
- Initial release!
