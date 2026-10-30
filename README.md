# wgpu-depth-pyramid

A small Rust / WGSL crate that builds a full min, max or AO-filtered depth mip chain from an
R32Float or Depth32Float texture. Output is R32Float. Requires Rust 1.88 and
`wgpu = "30"`; no other direct dependencies, optional GPU features, or unsafe code. The checked-in lockfile and MSRV-aware resolver
configuration keep transitive dependencies compatible with Rust 1.88.

## Usage

```rust,no_run
use wgpu_depth_pyramid::{PyramidKernel, Reduction, Viewport};

fn setup(device: &wgpu::Device, source: &wgpu::Texture) -> wgpu_depth_pyramid::Pyramid {
    let kernel = PyramidKernel::new(device); // Cache this across resizes/sources.
    kernel.prepare(device, source, Viewport::new([0, 0], [source.width(), source.height()]), Reduction::Min).unwrap()
}

fn frame(pyramid: &wgpu_depth_pyramid::Pyramid, encoder: &mut wgpu::CommandEncoder) {
    pyramid.record(encoder);
    // Sample pyramid.texture() or pyramid.mip_views() in subsequent passes.
}
```

`PyramidKernel::new` compiles pipelines for both formats, all reductions, and all
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

For min/max, depth values must be finite, nonnegative encodings in `[0,1]`, including positive
zero and positive subnormals. Negative zero, negatives, infinities and NaNs are
outside the contract. The GPU does no expensive input scan. Reduction compares
unsigned float bit patterns and reinterprets the selected value for storage;
this avoids floating arithmetic that can flush subnormals. Bit-exact agreement,
including subnormals, is tested on the qualification adapter. The [WGSL floating point rules](https://www.w3.org/TR/WGSL/#floating-point-evaluation)
permit flushing subnormal intermediate values during bit reinterpretation; shader
validation alone cannot promise bit preservation on every browser/driver.

## Depth-aware AO downsample (0.2.0)

Use `Reduction::Ao { effect_radius: 0.5, effect_falloff_range: 0.615 }` for a
reverse-Z closeness-weighted filter for screen-space ambient occlusion. Larger
values mean nearer samples. Radius uses the same units as the source depth;
this filter does not linearize depth. It is a weighted downsample rather than
a conservative surface bound.

For four depths `d0,d1,d2,d3` in raster order, the exact expression order is:

```text
m = max(max(d0,d1), max(d2,d3))
r = 0.75 * effect_radius
range = max(effect_falloff_range * r, 1e-5)
from = r * (1.0 - effect_falloff_range)
mul = -1.0 / range
add = from / range + 1.0
wi = clamp((m - di) * mul + add, 0.0, 1.0)
sum = ((w0 + w1) + w2) + w3
inv = if sum < 1e-5 { 0.0 } else { 1.0 / sum }
result = (((w0*d0 + w1*d1) + w2*d2) + w3*d3) * inv
```

Every operation and stored mip uses **f32** precision. The implementation keeps
the reference expression order, with no explicit fused multiply-add. Each fused
stage performs the same four-input filter; binary min/max-style combining would
change the result. AO arithmetic on the qualification adapter flushes subnormal
operands and results to signed zero. The CPU oracle models this f32 **FTZ**
precision explicitly at each arithmetic operation, without relying on host
FTZ flags. Mip-zero copies preserve the original encodings.
WGSL permits implementation-dependent rounding and subnormal handling, so
bitwise qualification applies to the tested adapter, rather than all WebGPU
implementations. An adapter preserving arithmetic subnormals may differ from
the FTZ CPU oracle; shader validation alone does not establish bitwise identity.

Dimensions still floor-halve to at least one. Cell `(x,y)` reads `(2x,2y)`,
`(min(2x+1,src_width-1),2y)`, `(2x,min(2y+1,src_height-1))` and the doubly
clamped fourth cell. Thin axes repeat samples. **Unmatched odd rows and columns
are dropped**, as in the per-mip reference filter: 5 -> 2 reads `[0,1]` and
`[2,3]`; 3 -> 1 reads `[0,1]`. Min/max retains its separate folding rule.
Mip zero is always a complete, unfiltered viewport copy, even for merged paths.

AO source depths use the same finite nonnegative `[0,1]` input domain, excluding
negative zero. Unlike min/max, AO performs floating-point arithmetic rather than
selecting an unchanged input encoding. Radius must be finite and nonnegative;
falloff must be finite in `[0,1]`.
Configurations whose derived coefficients overflow are rejected. Recreate the
prepared pyramid when either value changes; recording performs no parameter
uploads. AO supports both input formats and every strategy, including fused
staging within WebGPU default limits.

`Reduction`, `Strategy`, `Viewport`, `Error` and `reference::Mip` are now
`#[non_exhaustive]`. Match enums with a fallback arm and construct viewports with
`Viewport::new(offset, size)`. These are source-breaking changes in 0.2.0.

### AO precision and comparison bound

The independent CPU reference retains correctly rounded f32 FTZ arithmetic.
The AO bit-exact gate compares the GPU with the faithfully ported per-mip
reference shader **on the same adapter**, at every mip. Min/max remains bit-exact
against the CPU, and AO mip zero remains bit-exact against both references.

[WGSL division accuracy is 2.5 ULP per division](https://www.w3.org/TR/WGSL/#floating-point-accuracy)
for normal denominators in `[2^-126, 2^126]`. This is not a 2.5-ULP bound on the
final depth: errors in `-1/range`, `from/range`, and `1/sum` affect weights and
normalization, and prior-mip errors affect later quads. CPU comparison uses a
forward interval enclosure of the complete ordered formula:

- Start from exact input intervals `[d,d]`. Define `u(I)` as the largest f32
  spacing at the magnitude of either interval endpoint, at least `2^-149`.
- Evaluate interval addition/subtraction/product by their endpoint extrema,
  then widen by `u(I)` for either WGSL rounding direction. Evaluate quotient
  extrema and widen by **`2.5*u(I)`**. For four-term nonnegative sums use
  `e = 2^-23`, `gamma3 = 3e/(1-3e)`, and widen by
  `gamma3*upper_sum + 3*2^-126/(1-3e)`. This covers three directed-rounding
  additions in any order, including intermediate FTZ and binade crossings.
- Include zero wherever an operand or result can be subnormal; propagate
  `max` and `clamp` by their endpoint images. A nearest sample has weight
  `clamp(add,0,1)`, so the sum is at least `clamp(lower(add),0,1)`, near one
  for the qualified settings. This proves the zero normalization branch
  unreachable for those settings.
- Feed each output interval into the next mip. For CPU output `c` and its
  interval `[L,U]`, check `abs(gpu-c) <= max(c-L,U-c)` and require `c` inside
  the interval. f64 calculations of bounds round outward with `next_down/up`.

The result is also intersected with the weighted-average input hull, enlarged
by rounding. This prevents the independent numerator and denominator intervals
from producing an unbounded depth envelope at later mips. With `t=2^-126`,
`gamma_n=ne/(1-ne)`, `S0=clamp(lower(add),0,1)`, and endpoint input extrema:

```text
ES = gamma3 + 3t / ((1-3e)*S0)
EN = 7t / ((1-4e)*S0)
Klo = (1-2.5e)*(1-e) / (1+ES)
Khi = (1+2.5e)*(1+e) / (1-ES)
Lhull = min(d)*(1-gamma4)*Klo - EN*Khi - t
Uhull = max(d)*(1+gamma4)*Khi + EN*Khi + t
```

`gamma4` covers rounded products and numerator additions, `ES` denominator
additions, `2.5e` the division, and `e` the final multiplication. The absolute
terms cover intermediate and output FTZ. Intersect `[L,U]` with
`[max(0,Lhull),Uhull]` before passing it to the next mip. All bound calculations
round outward. A constant 13-level chain regression requires a final absolute
bound below `1e-4`, including zero-radius filtering.

The CPU-only implementation is in `examples/support/ao_bounds.rs`, shared by
qualification tests and benchmarks. It uses source values, parameters and mip
shapes; no GPU output or adapter-derived lookup data determines a tolerance.
A reciprocal regression fixture exercises the former two-bit discrepancy and
also checks that an incorrect unweighted average lies outside the bound.
The bound is conservative and can widen with sharp falloffs and later mips;
the simultaneous exact GPU-reference gate detects shader changes even then.
This enclosure covers the ordered formula and ordinary multiply-add contraction,
not every arbitrary reassociation a compiler might choose. Overflowing arithmetic
and denominators outside WGSL's specified precision range are not covered by
this numerical bound. Runtime identity is qualified on the tested adapter;
shader validation alone does not establish cross-adapter identity.

GPU-reference identity passed on the qualification adapter: **137,480 full
chains and 2,349,199,800 texels**, including all 61 sizes, both source formats,
both viewport origins, ten patterns, all explicit strategies at small sizes,
and additional radius/falloff extremes. The same matrix checks the independent
CPU reference with the bound above. The largest observed absolute CPU/GPU
difference was `7.1525574e-7`, and the largest error divided by its own bound
was `0.208`. The most conservative late-mip bound reached `0.976` for this
matrix; such envelopes can approach the depth range and are not estimates of
expected error. The simultaneous GPU-reference bitwise gate remains mandatory.

## Strategies and portability

`prepare` uses `Strategy::Auto`. `prepare_with_strategy` accepts explicit
`Naive`, `Merged`, `Fused { tile, depth }`, `FusedMerged { tile, depth }`, and `Legacy` choices for profiling
or adapter-specific tuning. `Pyramid::strategy()` reports the resolved choice.
Selection takes place at setup; recording never benchmarks or uploads parameters.

The automatic rule uses `FusedMerged { tile: 32, depth: 4 }` at 1920x1080,
2560x1440 and 3840x2160, for either source format and all reductions. All other
dimensions use `Naive`. Selection is static at setup. Automatic full chains use
**one compute pass**, four dispatches at these three resolutions, and no staged
copies. The previous 1080p depth-five choice required a buffer-to-texture copy;
it remains available explicitly but is no longer the default.

`record` rebinds the preallocated per-mip bind groups and pipelines before each
dispatch. WebGPU compute dispatches have individual usage scopes, so a mip can
be a storage output in one dispatch and a sampled input in the next. wgpu 30
merges bindings and emits required barriers at each dispatch, rather than
requiring a pass boundary. See the [WebGPU synchronization contract](https://www.w3.org/TR/webgpu/#programming-model-synchronization).
Our bindings never read and write an overlapping subresource in the same
dispatch, so they need no validation-conflict fallback.

`record_separate_passes` retains the per-dispatch pass shape for comparison.
`compute_pass_count`, `dispatch_count` and `staged_copy_count` report each cost
separately. Explicit depth-five/six strategies may stage extra levels through
buffers; those copies cannot occur inside a compute pass, so recording ends the
pass for a copy and resumes a new pass for its consumers. `record_profiled`
also uses separate passes for per-dispatch timestamp queries, without requiring
optional timestamp-inside-pass features. The default `record` path needs no
optional GPU features.

The scalar path uses an 8x8 workgroup, a mip-zero copy, and one dispatch per
reduction. `Merged` combines the copy and first reduction in registers with
256 invocations, followed by scalar dispatches. The tiled paths copy mip zero
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
reproducible comparisons. Dispatch-level synchronization establishes texture
visibility. No subgroup or atomic-counter path is enabled.

All paths use at most four write-only storage textures, one sampled texture,
one 48-byte uniform buffer, one storage buffer, 256 invocations per workgroup
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
exact min/max depth selection and a depth-aware AO filter; it does not offer
filtered color or the broader SPD options.

## Verification and timing

- `cargo test`: CPU coverage tests and Naga validation of every generated shader,
  with empty optional capabilities. The hardware integration test is marked
  `#[ignore]`, so the default suite never requests an adapter.
- `cargo test --test gpu -- --ignored --nocapture`: explicitly run the complete
  min/max bitwise hardware matrix. Run it under your machine's exclusive GPU scheduler.
  An unavailable adapter or CPU fallback fails this opt-in test, never skips it.
  Every source texture is independently read back before reducing. Depth fixtures
  use clears plus fragment discard to preserve subnormals across fragment-depth
  clamping: one subnormal encoding per depth texture, including separate smallest
  and largest subnormal cases and mixed normal/subnormal fields. R32Float fixtures
  also cover arbitrary random subnormal encodings.
- `cargo test --test ao -- --ignored --nocapture`: compare every AO mip against
  both the independent CPU oracle and the ported per-mip reference GPU shader.
  Run under an exclusive GPU scheduler. The existing 61-size matrix is reused.
  AO CPU comparisons use the propagated WGSL precision bound above; the
  ported GPU reference and mip-zero comparisons remain bit-exact.
- `cargo run --release --example encode`: compare one-pass and per-dispatch-pass
  CPU recording at 1080p with full Min, Max and AO chains per frame. Preallocate
  1,024/4,096/16,384 live storage buffers, use 512 in a shared frame prelude,
  and retain all buffers during measurement. Two hundred alternating pairs
  follow twenty warmup pairs. Report pyramid recording, complete encoding
  (encoder creation, prelude, pyramids and finish), and submission separately;
  polling is outside the CPU timings. Every mip must be bit-identical between
  recording modes before timing. This is a synthetic live-buffer workload, not
  a measured consumer scene.
- `cargo run --release --example bench -- --ao --final`: time automatic AO,
  the per-mip reference shader shape and the library scalar path in interleaved
  triples, with the same complete-chain timestamp boundaries. Before timing,
  every arm must match the original per-mip GPU reference bit-for-bit and the
  independent CPU reference within the propagated precision bound.
- `cargo build --target wasm32-unknown-unknown`: compile the WebGPU library path.
  This is a build gate, not a claim of browser runtime qualification.
- `cargo run --release --example bench`: run under an exclusive GPU scheduler.
  Requires the optional TIMESTAMP_QUERY feature **only for the benchmark**.
  Prints adapter/driver, per-frame raw GPU timestamps converted to microseconds,
  and medians for 40 interleaved triples after eight warmup triples at each resolution and
  reduction. Empty compute-pass timestamp markers surround the complete chain
  equally for all three arms. Setup, source upload and readback are outside timing.

The microbenchmark compares each candidate with the independent conventional
8x8 per-pass baseline and the library's optimized scalar dispatch path. The
0.2.0 default and scalar library arms each use one compute pass; the independent
per-mip reference arm retains separate passes. All arms preallocate
uniforms and bind groups. `-- --final` measures the automatic default. Forty
interleaved triples follow eight warmup triples; per-dispatch profiles also use
forty samples. Per-dispatch profiles use separate passes and cannot be added
up to reconstruct the one-pass chain timing. Raw GPU timestamps and CPU record
durations go to stderr.

GPU chain timestamps include dispatches, dependencies and staged mip copies.
CPU record medians exclude encoder creation, finish, submit, polling, timestamp
marker creation and readback. CPU and GPU costs must not be added as a frame
latency estimate. Steady-state bindings and parameters are cached in every arm. No consumer-inclusive
application speedup or universal cross-adapter ordering is implied.

## Performance

### 0.2.0 GPU chains

NVIDIA GeForce RTX 4080 Laptop GPU, Vulkan, NVIDIA driver 595.84, wgpu 30.0.1,
graphics clocks locked at 1500 MHz and reset afterward. Memory clocks were
adaptive. Values are median GPU microseconds from 40 interleaved triples after
eight warmup triples. All arms build the complete chain, including mip zero,
from the same synthetic R32Float input with cached uniforms and bindings.

Default and scalar library chains each use one compute pass; the independent
8x8 per-mip reference uses separate passes. The default uses
`FusedMerged { tile: 32, depth: 4 }`, four dispatches and no staging. Every arm
is verified against the CPU before timing (bitwise min/max, propagated AO
precision bound) and against the per-mip GPU reference bit-for-bit.

| Resolution | Reduction | Default one-pass µs | Per-mip reference µs | Scalar one-pass µs | Default / reference |
| --- | --- | ---: | ---: | ---: | ---: |
| 1920x1080 | Min | 48.128 | 115.200 | 61.440 | 0.4178 |
| 1920x1080 | Max | 48.128 | 114.688 | 60.416 | 0.4196 |
| 1920x1080 | AO | 40.960 | 107.520 | 59.392 | 0.3810 |
| 2560x1440 | Min | 73.728 | 160.768 | 95.232 | 0.4586 |
| 2560x1440 | Max | 73.728 | 160.768 | 95.232 | 0.4586 |
| 2560x1440 | AO | 67.584 | 153.600 | 94.720 | 0.4400 |
| 3840x2160 | Min | 233.984 | 333.312 | 277.504 | 0.7020 |
| 3840x2160 | Max | 266.240 | 368.128 | 295.936 | 0.7232 |
| 3840x2160 | AO | 296.960 | 424.448 | 369.152 | 0.6996 |

These are synthetic depth-chain measurements on one adapter. They do not
establish cross-adapter ordering or whole-frame application savings. Per-dispatch
profiles use separate passes and are not sums of the default chain timings.

### 0.2.0 CPU encoding at 1080p

AMD Ryzen 9 7945HX (16 cores / 32 threads, performance governor), native Vulkan
wgpu 30.0.1. The synthetic frame records full Min, Max and AO chains from a
Depth32Float source after a 128-dispatch buffer prelude; 512 pool buffers are
active, and every pool buffer remains alive. Automatic chains use 12 dispatches
in three passes versus twelve passes; scalar chains use 33 dispatches in three
passes versus 33 passes. Both arms reuse the same pyramids and bind groups.

Median CPU microseconds per frame from 200 alternating pairs after 20 warmup
pairs. Full encode includes encoder creation, prelude, pyramids and `finish`;
it captures deferred wgpu validation/tracker work that `record` alone misses.
Submission and GPU polling are outside full encode. GPU clocks were locked at
1500 MHz and reset afterward; CPU affinity was unrestricted. Shared-host load
and CPU frequency variation remain limitations.

| Live pool buffers | Strategy | One-pass full encode µs | Separate-pass full encode µs | One-pass pyramid record µs | Separate-pass pyramid record µs |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1,024 | Auto | 162.396 | 197.838 | 1.493 | 3.327 |
| 4,096 | Auto | 178.171 | 246.630 | 1.603 | 3.336 |
| 16,384 | Auto | 234.788 | 424.996 | 1.553 | 3.371 |
| 1,024 | Naive | 185.861 | 306.192 | 2.250 | 8.997 |
| 4,096 | Naive | 200.779 | 425.172 | 2.304 | 9.032 |
| 16,384 | Naive | 258.017 | 886.091 | 2.349 | 9.488 |

These are encoding costs for this synthetic frame, not isolated internal tracker
function timings or a consumer-inclusive frame-rate claim. Every output mip
is bit-identical between recording modes, including AO.

## Correctness coverage

The 0.2.0 one-pass default passed the min/max CPU-reference matrix: 4,880 chains
and 2,490,638,640 texel comparisons. Explicit variants added 14,280 chains and
350,317,800 comparisons. A focused 1080p test checked all mips in both recording
modes for both formats, all three reductions and Auto/Naive/Legacy strategies
(18 pairs), with no validation-conflict fallback. The encode workload also
checks every output mip between recording modes before timing. AO GPU-reference
identity and CPU tolerance coverage are described above.

The 0.1.0 min/max automatic policy passed 4,880 full chains and 2,490,638,640 bitwise comparisons
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
