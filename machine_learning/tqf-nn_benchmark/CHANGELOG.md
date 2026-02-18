# TQF-NN Benchmark Tools: Changelog

**We're fired up and forging some new experimental tools---it's a work in progress!**

All notable changes to these TQF-NN benchmark tools will be documented in this file. For some potential future advancements, see [`FUTURE_TODO.md`](FUTURE_TODO.md).

---

## [1.0.3] - 2026-02-18

### Added
- Additional automated tests for dual metrics, engine, and output formatters.

### Changed
- Tuned/enhanced TQF orbit mixing features for ℤ₆, D₆, and 𝕋₂₄.
- Fixed minor typos in output formatting/logging.

### Removed
- Some dead code. Winter cleaning.


## [1.0.2] - 2026-02-15

### Added
- Additional performance and caching optimizations to initial dataset loading/preprocessing to decrease the runtime from a few minutes to about 1 second.
- Performance optimization to simply displaying the usage help message. A bunch of unnecessary overhead was executing before the CLI parameters were even parsed, even when the user was just trying to display the CLI parameter usage help message, so this was reordered/optimized.

### Changed
- ℤ₆ data augmentation is now **disabled by default** (was enabled). CLI parameter changed from `--no-tqf-z6-augmentation` (opt-out) to `--z6-data-augmentation` (opt-in). Renamed to reflect that augmentation applies to all models (shared training DataLoader), not just TQF-ANN. This avoids conflicts with orbit mixing features and aligns with the finding that augmentation + orbit mixing hurt accuracy.

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
