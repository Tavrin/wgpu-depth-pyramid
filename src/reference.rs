//! CPU reference: exact min/max encodings and correctly rounded f32 FTZ AO arithmetic.
//! Inputs are nonnegative finite depth values in [0, 1].
use crate::{mip_sizes, validate, Error, Reduction, Viewport};

/// One tightly packed CPU mip level.
#[non_exhaustive]
#[derive(Debug)]
pub struct Mip {
    /// Width and height in texels.
    pub size: [u32; 2],
    /// Row-major depth values without padding.
    pub texels: Vec<f32>,
}

/// Build a tightly packed chain using the selected reduction.
/// Min/max selects raw unsigned encodings and preserves subnormals, so comparisons
/// with the GPU are bit-exact. AO uses ordered, correctly rounded f32 arithmetic
/// with explicit flush-to-zero (FTZ), clamped 2x2 samples and dropped odd tails.
/// Mip zero is always bit-exact, including subnormal encodings.
///
/// # AO comparison bound
///
/// AO GPU division need not equal the CPU division. WGSL permits 2.5 ULP per
/// division for normal denominators in `[2^-126, 2^126]`. The qualification
/// tests propagate intervals through the complete formula (including both
/// coefficient divisions and the normalization reciprocal) and through every
/// subsequent mip, starting with exact source intervals. For interval `I`, let
/// `u(I)` be the largest f32 spacing at either endpoint's magnitude (at least
/// `2^-149`). Interval `+`, `-`, `*` widen by `u(I)` for directed rounding;
/// interval `/` widens by `2.5*u(I)`. Four-term nonnegative sums use
/// `gamma3 = 3e/(1-3e)`, `e = 2^-23`: absolute error at most
/// `gamma3*upper_sum + 3*2^-126/(1-3e)` for three additions in any order,
/// including intermediate FTZ. Each operation encloses subnormal flushing and the
/// host f64 bound calculation is rounded outward. Max/clamp use endpoint images.
/// A nearest sample has weight `clamp(add,0,1)`, so the weight sum is at
/// least `clamp(lower_bound(add),0,1)` (near one for the qualified settings).
///
/// To avoid dependency-driven interval explosion, also intersect each result
/// with its input hull enlarged by normalization rounding. Define `t=2^-126`,
/// `gamma_n=ne/(1-ne)`, `S0=clamp(lower_bound(add),0,1)`,
/// `ES=gamma3+3t/((1-3e)S0)`, `EN=7t/((1-4e)S0)`, and
/// `K-=(1-2.5e)(1-e)/(1+ES)`, `K+=(1+2.5e)(1+e)/(1-ES)`.
/// The extra enclosure is `[min(d)*(1-gamma4)*K- - EN*K+ - t,
/// max(d)*(1+gamma4)*K+ + EN*K+ + t]`, clipped to nonnegative values.
/// Here `min(d)` and `max(d)` use the lower and upper input endpoints.
/// `gamma4` covers products plus numerator additions, `ES` denominator
/// additions, `2.5e` the reciprocal, and `e` final multiplication. The FTZ
/// terms cover four products, three additions and the final output.
///
/// For a final interval `[L,U]` and CPU value `c`, the absolute comparison bound
/// is `max(c-L, U-c)`. It is input/parameter/mip dependent, not 2.5 ULP of the
/// final depth. The CPU value must itself lie in the interval. Tests implement
/// this independently in `examples/support/ao_bounds.rs`, using no GPU-derived
/// lookup data. This enclosure qualifies the ordered formula and ordinary
/// multiply-add contraction; arbitrary compiler reassociation across the
/// formula or overflowing intermediates are not a cross-adapter guarantee.
/// GPU-versus-ported-reference-shader identity on the same adapter remains the
/// bit-exact AO gate. See the crate documentation for the formula and coverage.
///
/// Negative zero, negative values, infinities and NaNs are outside the input contract.
pub fn build(
    source: &[f32],
    source_size: [u32; 2],
    viewport: Viewport,
    reduction: Reduction,
) -> Result<Vec<Mip>, Error> {
    validate(source_size, viewport)?;
    reduction.validate()?;
    if u64::from(source_size[0]) * u64::from(source_size[1]) != source.len() as u64 {
        return Err(Error("source length does not match dimensions"));
    }
    if source.iter().any(|v| v.to_bits() > 1.0f32.to_bits()) {
        return Err(Error(
            "depth must be finite, nonnegative and at most one; negative zero is excluded",
        ));
    }
    let sizes = mip_sizes(viewport.size)?;
    let mut base = Vec::with_capacity(viewport.size[0] as usize * viewport.size[1] as usize);
    for y in viewport.offset[1]..viewport.offset[1] + viewport.size[1] {
        let start = y as usize * source_size[0] as usize + viewport.offset[0] as usize;
        base.extend_from_slice(&source[start..start + viewport.size[0] as usize]);
    }
    let mut mips = vec![Mip {
        size: viewport.size,
        texels: base,
    }];
    for &size in &sizes[1..] {
        let parent = mips.last().unwrap();
        let mut texels = Vec::with_capacity(size[0] as usize * size[1] as usize);
        for y in 0..size[1] {
            let end_y = if y + 1 == size[1] {
                parent.size[1]
            } else {
                2 * y + 2
            };
            for x in 0..size[0] {
                if let Reduction::Ao {
                    effect_radius,
                    effect_falloff_range,
                } = reduction
                {
                    let sx = 2 * x;
                    let sy = 2 * y;
                    let bx = (sx + 1).min(parent.size[0] - 1);
                    let by = (sy + 1).min(parent.size[1] - 1);
                    let get = |x, y| parent.texels[(y * parent.size[0] + x) as usize];
                    texels.push(filter(
                        [get(sx, sy), get(bx, sy), get(sx, by), get(bx, by)],
                        effect_radius,
                        effect_falloff_range,
                    ));
                    continue;
                }
                let end_x = if x + 1 == size[0] {
                    parent.size[0]
                } else {
                    2 * x + 2
                };
                let mut value = parent.texels[(2 * y * parent.size[0] + 2 * x) as usize].to_bits();
                for sy in 2 * y..end_y {
                    for sx in 2 * x..end_x {
                        let sample = parent.texels[(sy * parent.size[0] + sx) as usize].to_bits();
                        value = match reduction {
                            Reduction::Min => value.min(sample),
                            Reduction::Max => value.max(sample),
                            Reduction::Ao { .. } => unreachable!(),
                        };
                    }
                }
                texels.push(f32::from_bits(value));
            }
        }
        mips.push(Mip { size, texels });
    }
    Ok(mips)
}

// Independent scalar implementation of the reference expression order. Model
// the qualification GPU's f32 flush-to-zero (FTZ) arithmetic explicitly, without
// relying on the host floating-point environment. Mip-zero copies never use FTZ.
fn ftz(v: f32) -> f32 {
    if v.is_subnormal() {
        0.0f32.copysign(v)
    } else {
        v
    }
}
fn add(a: f32, b: f32) -> f32 {
    ftz(ftz(a) + ftz(b))
}
fn sub(a: f32, b: f32) -> f32 {
    ftz(ftz(a) - ftz(b))
}
fn mul(a: f32, b: f32) -> f32 {
    ftz(ftz(a) * ftz(b))
}
fn div(a: f32, b: f32) -> f32 {
    ftz(ftz(a) / ftz(b))
}

fn filter(d: [f32; 4], radius: f32, falloff: f32) -> f32 {
    let d = d.map(ftz);
    let nearest = d[0].max(d[1]).max(d[2].max(d[3]));
    let radius = mul(0.75, radius);
    let range = mul(falloff, radius).max(1e-5);
    let from = mul(radius, sub(1.0, falloff));
    let mul_factor = div(-1.0, range);
    let add_factor = add(div(from, range), 1.0);
    let w = d.map(|v| add(mul(sub(nearest, v), mul_factor), add_factor).clamp(0.0, 1.0));
    let sum = add(add(add(w[0], w[1]), w[2]), w[3]);
    let inv = if sum < 1e-5 { 0.0 } else { div(1.0, sum) };
    mul(
        add(
            add(add(mul(w[0], d[0]), mul(w[1], d[1])), mul(w[2], d[2])),
            mul(w[3], d[3]),
        ),
        inv,
    )
}
