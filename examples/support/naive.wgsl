struct Params { src_size: vec2<u32>, size: vec2<u32> }
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var output: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.xy;
    if any(p >= params.size) { return; }
    if COPY_BASE {
        textureStore(output, vec2<i32>(p), textureLoad(source, vec2<i32>(p), 0));
        return;
    }
    let start = min(p * 2u, params.src_size - 1u);
    let end = select(min(start + 2u, params.src_size), params.src_size, p + 1u == params.size);
    var value = bitcast<u32>(textureLoad(source, vec2<i32>(start), 0).r);
    for (var y = start.y; y < end.y; y++) {
        for (var x = start.x; x < end.x; x++) {
            let sample = bitcast<u32>(textureLoad(source, vec2<i32>(vec2<u32>(x, y)), 0).r);
            value = REDUCE(value, sample);
        }
    }
    textureStore(output, vec2<i32>(p), vec4<f32>(bitcast<f32>(value)));
}
