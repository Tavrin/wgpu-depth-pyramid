//! Independent CPU-only forward error enclosure for the ordered AO formula.
//! No adapter queries, shader outputs or fitted constants enter this calculation.
use wgpu_depth_pyramid::{reference, Reduction};

#[derive(Clone, Copy, Debug)]
pub struct Bounds(pub f64, pub f64);
impl Bounds {
    fn point(v: f32) -> Self {
        Self(f64::from(v), f64::from(v))
    }
    // WGSL can flush subnormal operands/results. Include both possibilities.
    fn ftz(self) -> Self {
        let tiny = f64::from(f32::MIN_POSITIVE);
        Self(
            if self.0 > 0.0 && self.0 < tiny {
                0.0
            } else {
                self.0
            },
            if self.1 < 0.0 && self.1 > -tiny {
                0.0
            } else {
                self.1
            },
        )
    }
    // One ULP covers either directed rounding for +, -, *; division allows
    // 2.5 ULP of its exact quotient. ULP uses the larger binade at boundaries.
    // f64 next_down/up encloses host arithmetic and the allowance calculation.
    fn rounded(self, ulps: f64) -> Self {
        let magnitude = self.0.abs().max(self.1.abs());
        assert!(magnitude.is_finite() && magnitude <= f64::from(f32::MAX));
        let value = magnitude as f32;
        let unit = f64::from(value.next_up()) - f64::from(value);
        let error = (ulps * unit).next_up();
        Self((self.0 - error).next_down(), (self.1 + error).next_up()).ftz()
    }
    fn add(self, rhs: Self) -> Self {
        let (a, b) = (self.ftz(), rhs.ftz());
        Self((a.0 + b.0).next_down(), (a.1 + b.1).next_up()).rounded(1.0)
    }
    fn sub(self, rhs: Self) -> Self {
        self.add(Self(-rhs.1, -rhs.0))
    }
    fn mul(self, rhs: Self) -> Self {
        let (a, b) = (self.ftz(), rhs.ftz());
        let p = [a.0 * b.0, a.0 * b.1, a.1 * b.0, a.1 * b.1];
        Self(
            p.into_iter().fold(f64::INFINITY, f64::min).next_down(),
            p.into_iter().fold(f64::NEG_INFINITY, f64::max).next_up(),
        )
        .rounded(1.0)
    }
    fn div(self, rhs: Self) -> Self {
        let (a, b) = (self.ftz(), rhs.ftz());
        // WGSL's 2.5-ULP guarantee requires a normal denominator <= 2^126.
        assert!(b.0 >= f64::from(f32::MIN_POSITIVE) && b.1 <= 2.0f64.powi(126));
        let p = [a.0 / b.0, a.0 / b.1, a.1 / b.0, a.1 / b.1];
        Self(
            p.into_iter().fold(f64::INFINITY, f64::min).next_down(),
            p.into_iter().fold(f64::NEG_INFINITY, f64::max).next_up(),
        )
        .rounded(2.5)
    }
    fn max(self, rhs: Self) -> Self {
        Self(self.0.max(rhs.0), self.1.max(rhs.1))
    }
    fn clamp(self) -> Self {
        Self(self.0.clamp(0.0, 1.0), self.1.clamp(0.0, 1.0))
    }
    fn sum(values: [Self; 4]) -> Self {
        // For three additions of nonnegative terms in any order, directed
        // rounding has relative error <= gamma3 = 3e/(1-3e), e=2^-23.
        // Each intermediate FTZ can additionally lose MIN_POSITIVE. Using
        // gamma3 also covers a rounded partial sum crossing a binade boundary.
        let values = values.map(Self::ftz);
        let low = values.iter().fold(0.0, |s, v| (s + v.0).next_down());
        let high = values.iter().fold(0.0, |s, v| (s + v.1).next_up());
        let denominator = 1.0 - 3.0 * f64::from(f32::EPSILON);
        let rounding = (3.0 * f64::from(f32::EPSILON) * high.max(0.0)).next_up();
        let numerator = (rounding + 3.0 * f64::from(f32::MIN_POSITIVE)).next_up();
        let error = (numerator / denominator).next_up();
        Self((low - error).next_down().max(0.0), (high + error).next_up()).ftz()
    }
    pub fn tolerance(self, cpu: f32) -> f64 {
        let cpu = f64::from(cpu);
        assert!(cpu >= self.0 && cpu <= self.1, "CPU {cpu} outside {self:?}");
        let error = (cpu - self.0).max(self.1 - cpu);
        if error == 0.0 {
            0.0
        } else {
            error.next_up()
        }
    }
}

// Tighten the generic expression interval with N/S lying in the input hull.
// Products + three numerator additions have gamma4 relative error and at
// most 7*t/(1-4e) absolute FTZ error. The denominator's three additions have
// gamma3 relative error and 3*t/(1-3e) absolute FTZ error. Division contributes
// 2.5e relative error, final multiplication e, and final FTZ at most t.
// All factors are derived here, independently of any observed GPU error.
fn average_gains(sum_lower: f64) -> (Bounds, f64) {
    let e = f64::from(f32::EPSILON);
    let t = f64::from(f32::MIN_POSITIVE);
    let gamma = |n: f64| (n * e / (1.0 - n * e)).next_up();
    let numerator_ftz = ((7.0 * t / (1.0 - 4.0 * e)).next_up() / sum_lower).next_up();
    let denominator_ftz = ((3.0 * t / (1.0 - 3.0 * e)).next_up() / sum_lower).next_up();
    let denominator_error = (gamma(3.0) + denominator_ftz).next_up();
    assert!(sum_lower > 0.0 && denominator_error < 1.0);
    let low = ((1.0 - 2.5 * e) * (1.0 - e)).next_down();
    let high = ((1.0 + 2.5 * e) * (1.0 + e)).next_up();
    let low = (low / (1.0 + denominator_error).next_up()).next_down();
    let high = (high / (1.0 - denominator_error).next_down()).next_up();
    let ftz = ((numerator_ftz * high).next_up() + t).next_up();
    (
        Bounds(
            (low * (1.0 - gamma(4.0)).next_down()).next_down(),
            (high * (1.0 + gamma(4.0)).next_up()).next_up(),
        ),
        ftz,
    )
}

pub fn build(cpu: &[reference::Mip], reduction: Reduction) -> Vec<Vec<Bounds>> {
    let Reduction::Ao {
        effect_radius,
        effect_falloff_range,
    } = reduction
    else {
        panic!("AO bounds require AO reduction")
    };
    let one = Bounds::point(1.0);
    let radius = Bounds::point(0.75).mul(Bounds::point(effect_radius));
    let falloff = Bounds::point(effect_falloff_range);
    let range = falloff.mul(radius).max(Bounds::point(1e-5));
    let from = radius.mul(one.sub(falloff));
    let mul = Bounds::point(-1.0).div(range);
    let add = from.div(range).add(one);
    let sum_lower = add.0.clamp(0.0, 1.0);
    let (gains, average_ftz) = average_gains(sum_lower);
    let mut chain = vec![cpu[0]
        .texels
        .iter()
        .map(|&v| Bounds::point(v))
        .collect::<Vec<_>>()];
    for level in 1..cpu.len() {
        let parent = &chain[level - 1];
        let [pw, ph] = cpu[level - 1].size;
        let [w, h] = cpu[level].size;
        let mut next = Vec::with_capacity(cpu[level].texels.len());
        for y in 0..h {
            for x in 0..w {
                let bx = (2 * x + 1).min(pw - 1);
                let by = (2 * y + 1).min(ph - 1);
                let d = [(2 * x, 2 * y), (bx, 2 * y), (2 * x, by), (bx, by)]
                    .map(|(x, y)| parent[(y * pw + x) as usize].ftz());
                let m = d[0].max(d[1]).max(d[2].max(d[3]));
                let weights = d.map(|v| m.sub(v).mul(mul).add(add).clamp());
                let mut sum = Bounds::sum(weights);
                // For a nearest sample m-di is exactly zero, so its weight
                // is clamp(add,0,1). Retain that dependency even though m's
                // generic interval loses correlation with each input interval.
                // Use add's lower endpoint rather than assuming approximate
                // division of zero must return exact zero on every adapter.
                sum.0 = sum.0.max(sum_lower);
                let inv = one.div(sum);
                let products = std::array::from_fn(|i| weights[i].mul(d[i]));
                let result = Bounds::sum(products).mul(inv);
                let min = d.into_iter().map(|v| v.0).fold(f64::INFINITY, f64::min);
                let max = d.into_iter().map(|v| v.1).fold(0.0, f64::max);
                let low = ((min * gains.0).next_down() - average_ftz).next_down();
                let high = ((max * gains.1).next_up() + average_ftz).next_up();
                next.push(Bounds(result.0.max(low).max(0.0), result.1.min(high)));
            }
        }
        chain.push(next);
    }
    chain
}
