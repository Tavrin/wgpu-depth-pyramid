# Changelog

## [0.2.0] - 2026-10-30

### Added

- Parameterized `Reduction::Ao` for a reverse-Z depth-aware weighted 2x2
  downsample, with faithful f32 shader arithmetic and clamped thin edges.
  The CPU oracle models the qualification adapter's flush-to-zero arithmetic;
  mip-zero copies preserve the original encodings.
- AO support in scalar, merged, legacy and fused paths, including staged levels
  and per-resolution automatic selection within WebGPU default limits.
- AO bit-exact per-mip GPU reference tests and independent CPU comparison
  with a forward error bound derived from WGSL's 2.5 ULP division accuracy,
  propagated through the formula and successive mips; no GPU-derived lookup data.
  Min/max CPU comparisons remain bit-exact.
- AO GPU benchmarks and one-pass versus separate-pass CPU encoding benchmarks.
- `Viewport::new(offset, size)` constructor.

### Changed

- Default recording places all mip dispatches in one compute pass; wgpu 30
  tracks compute usages and inserts barriers per dispatch. Bind groups remain
  preallocated. Added `record_separate_passes` and `compute_pass_count`.
- Automatic 1080p fusion depth is four, avoiding staged copies so the complete
  chain fits one compute pass. Explicit staged strategies split at copies;
  per-dispatch timestamp profiling uses separate passes.
- Added a 1080p CPU encode comparison with a live-buffer pool and frame prelude.
- Breaking: `Reduction`, `Strategy`, `Viewport`, `Error` and `reference::Mip`
  are non-exhaustive; external viewport literals must use the constructor.
- Breaking: `Reduction` implements `PartialEq` rather than `Eq` because AO
  parameters are floating-point values.
- Parameter buffers are 48 bytes to hold AO settings, allocated only at setup.
- Public-text scan also excludes the `.git` file used by linked worktrees.

## [0.1.0] - 2026-10-30

- Build full R32Float min/max depth pyramids from R32Float or Depth32Float sources.
- Support cropped viewports and odd-size folding without losing edge texels.
- Provide scalar, merged and fused WGSL paths within WebGPU default limits.
- Cache pipelines and bindings; record chains without host allocations or uploads.
- Include an independent CPU reference, bitwise GPU tests and timestamp benchmarks.
