// Independent per-mip reference port: clamped 2x2, original f32 expression order.
struct WeightedParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    // AO radius and falloff parameters, shared with the sampling algorithm.
    effect_radius: f32,
    effect_falloff_range: f32,
    offset_x: u32,
    offset_y: u32,
};

@group(0) @binding(0) var src_mip: texture_2d<f32>;
@group(0) @binding(1) var dst_mip: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: WeightedParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dst_x = global_id.x;
    let dst_y = global_id.y;
    if dst_x >= params.dst_width || dst_y >= params.dst_height {
        return;
    }

    if COPY_BASE {
        let value = textureLoad(src_mip, vec2<i32>(i32(dst_x + params.offset_x), i32(dst_y + params.offset_y)), 0)COMPONENT;
        textureStore(dst_mip, vec2<i32>(i32(dst_x), i32(dst_y)), vec4<f32>(value));
        return;
    }

    // Source texel coordinates (2:1 reduction).
    let src_x = i32(dst_x * 2u);
    let src_y = i32(dst_y * 2u);
    let max_x = i32(params.src_width - 1u);
    let max_y = i32(params.src_height - 1u);

    // Load 2x2 block from source mip, clamping at edges for odd dimensions.
    let d0 = textureLoad(src_mip, vec2<i32>(src_x,                    src_y),                    0)COMPONENT;
    let d1 = textureLoad(src_mip, vec2<i32>(min(src_x + 1, max_x),    src_y),                    0)COMPONENT;
    let d2 = textureLoad(src_mip, vec2<i32>(src_x,                    min(src_y + 1, max_y)),    0)COMPONENT;
    let d3 = textureLoad(src_mip, vec2<i32>(min(src_x + 1, max_x),    min(src_y + 1, max_y)),    0)COMPONENT;

    // XeGTAO weighted-average filter. In reverse-Z, "closer to camera" = larger
    // depth value, so max_depth is the nearest sample. Weights peak at max_depth
    // and fall off smoothly over an AO-radius-scaled window; samples beyond the
    // window contribute zero, samples within contribute linearly.
    let max_depth = max(max(d0, d1), max(d2, d3));

    let depth_range_scale_factor = 0.75;
    let effect_radius = depth_range_scale_factor * params.effect_radius;
    let falloff_range = max(params.effect_falloff_range * effect_radius, 1e-5);
    let falloff_from  = effect_radius * (1.0 - params.effect_falloff_range);
    let falloff_mul   = -1.0 / falloff_range;
    let falloff_add   = falloff_from / falloff_range + 1.0;

    let w0 = saturate((max_depth - d0) * falloff_mul + falloff_add);
    let w1 = saturate((max_depth - d1) * falloff_mul + falloff_add);
    let w2 = saturate((max_depth - d2) * falloff_mul + falloff_add);
    let w3 = saturate((max_depth - d3) * falloff_mul + falloff_add);

    let weight_sum = w0 + w1 + w2 + w3;
    let inv_weight = select(1.0 / weight_sum, 0.0, weight_sum < 1e-5);
    let result = (w0 * d0 + w1 * d1 + w2 * d2 + w3 * d3) * inv_weight;

    textureStore(dst_mip, vec2<i32>(i32(dst_x), i32(dst_y)), vec4<f32>(result, 0.0, 0.0, 0.0));
}
