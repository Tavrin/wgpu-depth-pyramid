use crate::Reduction;

// Generate binding declarations and unroll the four workgroup-local stages at setup.
// No shader generation takes place while recording a frame.
pub(crate) fn generate(depth: bool, copy: bool, reduction: Reduction, count: u32) -> String {
    let mut code = format!(
        r#"
struct Params {{ src_size: vec2<u32>, offset: vec2<u32>, first_size: vec2<u32>, pad: vec2<u32>, effect_radius: f32, effect_falloff_range: f32, pad2: vec2<u32> }}
@group(0) @binding(0) var source: {texture};
@group(0) @binding(1) var<uniform> params: Params;
// Ping-pong prevents a writer racing another invocation's reads within a stage.
var<workgroup> scratch: array<u32, 512>;
fn combine(a: u32, b: u32) -> u32 {{ return {op}(a, b); }}
fn load(p: vec2<u32>) -> u32 {{
    return bitcast<u32>(textureLoad(source, vec2<i32>(p + params.offset), 0){component});
}}
// Exact source interval, including the unmatched odd texel only at the last cell.
fn end_axis(p: u32, source_size: u32, dest_size: u32) -> u32 {{
    return select(min(2u * p + 2u, source_size), source_size, p + 1u == dest_size);
}}
"#,
        texture = if depth {
            "texture_depth_2d"
        } else {
            "texture_2d<f32>"
        },
        component = if depth { "" } else { ".r" },
        op = if reduction == Reduction::Min {
            "min"
        } else {
            "max"
        }
    );
    if reduction.is_ao() {
        code.push_str(crate::ao::QUAD);
    }
    for i in 0..count {
        code.push_str(&format!(
            "@group(0) @binding({}) var out{i}: texture_storage_2d<r32float, write>;\n",
            i + 2
        ));
    }
    code.push_str(&format!(
        r#"
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {{
    let tile = {tile}u;
    let groups = max(params.first_size / tile, vec2<u32>(1u));
    let base = group.xy * tile;
    // The last tile absorbs up to tile-1 extra cells. This also handles thin chains.
    let extent = select(vec2<u32>(tile), params.first_size - base, group.xy + 1u == groups);
    for (var i = lane; i < extent.x * extent.y; i += 64u) {{
        let p = vec2<u32>(i % extent.x, i / extent.x);
        let global = base + p;
"#,
        tile = 1 << (count - 1)
    ));
    if copy {
        code.push_str("        let value = load(global);\n");
    } else if reduction.is_ao() {
        code.push_str("let a=2u*global; let b=min(a+1u,params.src_size-1u); let value=quad(load(a),load(vec2<u32>(b.x,a.y)),load(vec2<u32>(a.x,b.y)),load(b));\n");
    } else {
        code.push_str(
            r#"        var value = load(2u * global);
        let end = vec2<u32>(end_axis(global.x, params.src_size.x, params.first_size.x),
                            end_axis(global.y, params.src_size.y, params.first_size.y));
        for (var y = 2u * global.y; y < end.y; y++) {
            for (var x = 2u * global.x; x < end.x; x++) {
                value = combine(value, load(vec2<u32>(x, y)));
            }
        }
"#,
        );
    }

    code.push_str(
        r#"        scratch[p.y * 16u + p.x] = value;
        textureStore(out0, vec2<i32>(global), vec4<f32>(bitcast<f32>(value)));
    }
"#,
    );
    for stage in 1..count {
        let previous = if stage % 2 == 1 { 0 } else { 256 };
        let next = 256 - previous;
        code.push_str(&format!(
            r#"
    workgroupBarrier();
    let prev{stage} = max(extent >> vec2<u32>({shift}u), vec2<u32>(1u));
    let size{stage} = max(extent >> vec2<u32>({stage}u), vec2<u32>(1u));
    for (var i = lane; i < size{stage}.x * size{stage}.y; i += 64u) {{
        let p = vec2<u32>(i % size{stage}.x, i / size{stage}.x);
        let start = min(2u * p, prev{stage} - 1u);
        let end = vec2<u32>(end_axis(p.x, prev{stage}.x, size{stage}.x),
                            end_axis(p.y, prev{stage}.y, size{stage}.y));
{reduce_shared}        scratch[{next}u + p.y * 16u + p.x] = value;
        textureStore(out{stage}, vec2<i32>((base >> vec2<u32>({stage}u)) + p), vec4<f32>(bitcast<f32>(value)));
    }}
"#,
            reduce_shared = if reduction.is_ao() {
                format!("let b=min(start+1u,prev{stage}-1u); let value=quad(scratch[{previous}u+start.y*16u+start.x],scratch[{previous}u+start.y*16u+b.x],scratch[{previous}u+b.y*16u+start.x],scratch[{previous}u+b.y*16u+b.x]);\n")
            } else {
                format!(r#"        var value = scratch[{previous}u + start.y * 16u + start.x];
        for (var y = start.y; y < end.y; y++) {{
            for (var x = start.x; x < end.x; x++) {{
                value = combine(value, scratch[{previous}u + y * 16u + x]);
            }}
        }}
"#)
            },
            shift = stage - 1
        ));
    }
    code.push_str("}\n");
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(source: &str) {
        let module = wgpu::naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
    }

    #[test]
    fn all_shaders_validate() {
        for (depth, copy) in [(false, true), (true, true), (false, false)] {
            for reduction in [Reduction::Min, Reduction::Max, Reduction::AO] {
                for count in 1..=4 {
                    let source = generate(depth, copy, reduction, count);
                    validate(&source);
                }
            }
        }
    }

    #[test]
    fn ao_reference_validates() {
        for copy in ["true", "false"] {
            let code = include_str!("../examples/support/ao_reference.wgsl")
                .replace("COPY_BASE", copy)
                .replace("COMPONENT", ".r");
            validate(&code);
            validate(
                &code
                    .replace("texture_2d<f32>", "texture_depth_2d")
                    .replace("0).r", "0)"),
            );
        }
    }

    #[test]
    fn baseline_shaders_validate() {
        for copy in ["true", "false"] {
            for reduce in ["min", "max"] {
                validate(
                    &include_str!("../examples/support/naive.wgsl")
                        .replace("COPY_BASE", copy)
                        .replace("REDUCE", reduce),
                );
            }
        }
    }
}
