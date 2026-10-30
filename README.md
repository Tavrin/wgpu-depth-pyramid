# wgpu-depth-pyramid

A small Rust / WGSL crate that builds a full min or max depth mip chain from an
R32Float or Depth32Float texture. Output is R32Float. Requires Rust 1.88 and
`wgpu = "30"`; no other direct dependencies, optional GPU features, or unsafe code. The checked-in lockfile and MSRV-aware resolver
configuration keep transitive dependencies compatible with Rust 1.88.

## Usage

```rust,no_run
use wgpu_depth_pyramid::{PyramidKernel, Reduction, Viewport};

fn setup(device: &wgpu::Device, source: &wgpu::Texture) -> wgpu_depth_pyramid::Pyramid {
    let kernel = PyramidKernel::new(device); // Cache this across resizes/sources.
    kernel.prepare(device, source, Viewport {
        offset: [0, 0], size: [source.width(), source.height()],
    }, Reduction::Min).unwrap()
}

fn frame(pyramid: &wgpu_depth_pyramid::Pyramid, encoder: &mut wgpu::CommandEncoder) {
    pyramid.record(encoder);
    // Sample pyramid.texture() or pyramid.mip_views() in subsequent passes.
}
```

`PyramidKernel::new` compiles pipelines for both formats, both reductions, and all
scalar, merged, legacy and tiled variants. `prepare` allocates the output texture, per-mip
views, immutable parameter buffers and bind groups. Call it again on resize or
when changing the source texture, viewport or reduction. The returned pyramid
retains its pipelines and bindings, so the kernel can also be dropped after setup.
Keep the output and source alive for repeated recordings; updates to the same
source texture are visible on the next recording. `record` records the entire
chain into a caller-owned encoder without host allocations, buffer uploads,
resource creation, or shader compilation in this crate. The encoder/backend may
itself allocate while recording commands.

The source must have TEXTURE_BINDING usage, be single-sampled and 2D; its base
mip and first array layer are read. The nonempty viewport must fit inside it.
The output is exactly viewport-sized and includes a copied mip zero, then every
floor-halved level down to 1x1. Its usages are TEXTURE_BINDING, STORAGE_BINDING,
and COPY_SRC. The viewport is a cropped independent pyramid, not an incremental
update to an atlas. Device ownership and device limits follow normal wgpu rules.

## Reduction and odd edges

Choose `Reduction::Min` or `Reduction::Max` explicitly. In **reverse-Z**, greater
depth is closer: **Min is the farthest surface** (conservative occlusion), while
**Max is the closest surface** (hierarchical ray marching). These represent
different queries and must not be substituted for each other. With ordinary-Z
the near/far interpretation reverses.

For each axis, `dst = max(floor(src / 2), 1)`. Destination cell `i` owns source
cells `[2*i, 2*i+2)`, except the final destination cell owns through `src`.
Thus 5 -> 2 maps `[0,1]` and `[2,3,4]`; 3 -> 1 covers all three; 1 -> 1 keeps
the single cell. The Cartesian product of those intervals gives a 1x1 through
3x3 footprint. This partitions the parent without skipping or sharing texels.
Viewport offsets apply only to the initial load, and the rule is applied again
at every level. `reference::build` implements the same contract independently
on the CPU; `mip_sizes` supplies the chain dimensions.

Depth values must be finite, nonnegative encodings in `[0,1]`, including positive
zero and positive subnormals. Negative zero, negatives, infinities and NaNs are
outside the contract. The GPU does no expensive input scan. Reduction compares
unsigned float bit patterns and reinterprets the selected value for storage;
this avoids floating arithmetic that can flush subnormals. Bit-exact agreement,
including subnormals, is tested on the qualification adapter. The [WGSL floating point rules](https://www.w3.org/TR/WGSL/#floating-point-evaluation)
permit flushing subnormal intermediate values during bit reinterpretation; shader
validation alone cannot promise bit preservation on every browser/driver.

## Strategies and portability

`prepare` uses `Strategy::Auto`. `prepare_with_strategy` accepts explicit
`Naive`, `Merged`, `Fused { tile, depth }`, `FusedMerged { tile, depth }`, and `Legacy` choices for profiling
or adapter-specific tuning. `Pyramid::strategy()` reports the resolved choice.
Selection takes place at setup; recording never benchmarks or uploads parameters.

The automatic rule uses `FusedMerged { tile: 32, depth: 5 }` at 1920x1080 and
`FusedMerged { tile: 32, depth: 4 }` at 2560x1440 or 3840x2160, for either source
format. All other dimensions use `Naive` until separately timed. The 1080p path
records three compute passes and one staged mip copy per chain; the other two
record four compute passes without staged copies. `dispatch_count()` and
`staged_copy_count()` expose both costs.
Both reductions use the same rule. This is a static policy qualified on an RTX
4080 Laptop GPU, Vulkan, driver 595.84 with a 1500 MHz graphics-clock lock;
resolution alone cannot guarantee the fastest path on every adapter. Explicit
strategy selection supports local qualification and a scalar override.

The scalar path uses an 8x8 workgroup, a mip-zero copy, and one pass per
reduction. `Merged` combines the copy and first reduction in registers with
256 invocations, followed by scalar passes. The tiled paths copy mip zero
with the scalar kernel, then dispatch **256 invocations over 32x32 or 64x64
source tiles**. The 32-wide path reduces each 2x2 source quad in registers.
The 64-wide path produces four first-level values and their parent from a
4x4 source footprint in registers, eliminating one shared-memory stage.
Remaining levels alternate two shared banks (5 KiB total) with uniform barriers.
At odd edges, the final tile owns the remainder, and reduction footprints follow
the same folding rule at every intermediate level. A tile is never allowed to
reduce across another workgroup's footprint.

Fusion depth is 2..=5 for a 32-wide source tile and 2..=6 for a 64-wide tile.
Five- and six-output experiments write their extra levels into a persistent
storage buffer, then copy them into the mip texture with aligned row pitches.
These copy costs are part of complete-chain timing. This preserves the default
limit of **four storage textures** rather than requesting higher native limits.
`FusedMerged` combines the merged initial copy/reduction with the tiled tail.
Small tails specialize to the remaining number of outputs. `Legacy` retains
the original 64-thread, four-output, shared-memory copy/reduction algorithm for
reproducible comparisons. Every dispatch is a separate compute pass for global
texture visibility. No subgroup or atomic-counter path is enabled.

All paths use at most four write-only storage textures, one sampled texture,
one 32-byte uniform buffer, one storage buffer, 256 invocations per workgroup
and 5 KiB workgroup storage. These fit **WebGPU default limits** with empty
optional features. The output also has COPY_DST usage for portable staged levels.
`record_profiled` optionally records two timestamp queries per dispatch; only
that profiling call requires TIMESTAMP_QUERY. Whole-chain timestamps must also
include staged buffer-to-texture copies when comparing deeper variants.

[AMD FidelityFX SPD](https://github.com/GPUOpen-Effects/FidelityFX-SPD) (MIT) is
an algorithm reference, not a code source: SPD's
[documented design](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/single-pass-downsampler/)
reduces 64x64 tiles to 1x1, with 256 threads per group as described in AMD's
[architecture presentation](https://www.gpuopen.com/wp-content/uploads/slides/GPUOpen_Let%E2%80%99sBuild2020_Optimizing%20for%20the%20Radeon%20RDNA%20Architecture.pdf).
This is an independently written
WGSL implementation. SPD offers a single-dispatch path with a global counter
and optional wave operations. Here dispatch boundaries provide portable global
visibility. A native single-dispatch atomic-counter tail is intentionally
omitted: an atomic counter alone does not establish device-wide visibility of
other workgroups' texture writes under the
[WGSL memory model](https://www.w3.org/TR/WGSL/#memory-semantics). This crate offers
only exact min/max depth selection, not filtered color, weighted AO or SPD's
broader options.

## Verification and timing

- `cargo test`: CPU coverage tests and Naga validation of every generated shader,
  with empty optional capabilities. The hardware integration test is marked
  `#[ignore]`, so the default suite never requests an adapter.
- `cargo test --test gpu -- --ignored --nocapture`: explicitly run the complete
  bitwise hardware matrix. Run it under your machine's exclusive GPU scheduler.
  An unavailable adapter or CPU fallback fails this opt-in test, never skips it.
  Every source texture is independently read back before reducing. Depth fixtures
  use clears plus fragment discard to preserve subnormals across fragment-depth
  clamping: one subnormal encoding per depth texture, including separate smallest
  and largest subnormal cases and mixed normal/subnormal fields. R32Float fixtures
  also cover arbitrary random subnormal encodings.
- `cargo build --target wasm32-unknown-unknown`: compile the WebGPU library path.
  This is a build gate, not a claim of browser runtime qualification.
- `cargo run --release --example bench`: run under an exclusive GPU scheduler.
  Requires the optional TIMESTAMP_QUERY feature **only for the benchmark**.
  Prints adapter/driver, per-frame raw GPU timestamps converted to microseconds,
  and medians for 40 interleaved triples after eight warmup triples at each resolution and
  reduction. Empty compute-pass timestamp markers surround the complete chain
  equally for all three arms. Setup, source upload and readback are outside timing.

The microbenchmark compares each candidate with the independent conventional
8x8 baseline and the library's optimized scalar path. All arms preallocate
uniforms and bind groups. `-- --final` measures the automatic default. Forty
interleaved triples follow eight warmup triples; per-dispatch profiles also use
forty samples. Raw GPU timestamps and CPU record durations go to stderr.

GPU chain timestamps include dispatches, dependencies and staged mip copies.
CPU record medians exclude encoder creation, finish, submit, polling, timestamp
marker creation and readback. CPU and GPU costs must not be added as a frame
latency estimate. Steady-state bindings and parameters are cached in every arm. No consumer-inclusive
application speedup or universal cross-adapter ordering is implied.

## Performance

Measured on one adapter: NVIDIA GeForce RTX 4080 Laptop GPU, Vulkan, NVIDIA
driver 595.84, graphics clocks locked at 1500 MHz. Memory clocks were adaptive.
These P1b measurements compare the automatic fused path with this crate's
optimized per-mip path (`Strategy::Naive`), including the mip-zero copy.
Values are median GPU microseconds from 40 interleaved triples after eight
warmup triples; the third arm was an independent conventional baseline.

| Resolution | Reduction | Fused µs | Crate per-mip µs | Fused / per-mip |
| --- | --- | ---: | ---: | ---: |
| 1920x1080 | Min | 60.416 | 110.592 | 0.5463 |
| 1920x1080 | Max | 61.440 | 111.616 | 0.5505 |
| 2560x1440 | Min | 89.088 | 154.112 | 0.5781 |
| 2560x1440 | Max | 90.624 | 155.136 | 0.5842 |
| 3840x2160 | Min | 279.552 | 356.352 | 0.7845 |
| 3840x2160 | Max | 315.392 | 390.656 | 0.8073 |

This is a synthetic depth-chain benchmark on one adapter. It does not establish
cross-adapter ordering or whole-frame application savings.

## Correctness coverage

The automatic policy passed 4,880 full chains and 2,490,638,640 bitwise comparisons
against the independent CPU reference. The matrix covers both source formats,
both reductions, ten patterns and 61 sizes: 1x1, degenerate axes up to 4097,
odd/even and tile boundaries through 257x129, 1080p, 1440p, 4K, 4097x2049,
and 32 seeded random rectangles up to 300x300. Each runs at origin and at
viewport offset `[3,5]`. Patterns include zero, one, equal depths, random depths,
edge extrema and subnormal encodings. Repeated recording after source updates
is checked too. The explicit-strategy suite adds 14,280 chains and 350,317,800
bitwise comparisons across all 21 strategies. CPU partition tests cover every
size from 1x1 through 65x65 with both reductions.

Native GPU tests request exactly default limits and no optional features.
Naga validates generated WGSL with all validation flags and empty optional
capabilities. A wasm32 build checks compilation; browser runtime and
cross-vendor execution have not been tested.

## Licence

Licensed under MIT OR Apache-2.0, at your option. See LICENSE-MIT and
LICENSE-APACHE.
