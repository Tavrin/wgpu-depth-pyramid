// Four samples in raster order. Keep the reference shader's expression order;
// this operation is not associative and must never be split into binary combines.
pub(crate) const QUAD: &str = r#"
fn quad(a: u32, b: u32, c: u32, d: u32) -> u32 {
    let d0 = bitcast<f32>(a);
    let d1 = bitcast<f32>(b);
    let d2 = bitcast<f32>(c);
    let d3 = bitcast<f32>(d);
    let max_depth = max(max(d0, d1), max(d2, d3));
    let depth_range_scale_factor = 0.75;
    let effect_radius = depth_range_scale_factor * params.effect_radius;
    let falloff_range = max(params.effect_falloff_range * effect_radius, 1e-5);
    let falloff_from = effect_radius * (1.0 - params.effect_falloff_range);
    let falloff_mul = -1.0 / falloff_range;
    let falloff_add = falloff_from / falloff_range + 1.0;
    let w0 = saturate((max_depth - d0) * falloff_mul + falloff_add);
    let w1 = saturate((max_depth - d1) * falloff_mul + falloff_add);
    let w2 = saturate((max_depth - d2) * falloff_mul + falloff_add);
    let w3 = saturate((max_depth - d3) * falloff_mul + falloff_add);
    let weight_sum = w0 + w1 + w2 + w3;
    let inv_weight = select(1.0 / weight_sum, 0.0, weight_sum < 1e-5);
    let result = (w0 * d0 + w1 * d1 + w2 * d2 + w3 * d3) * inv_weight;
    return bitcast<u32>(result);
}
"#;
