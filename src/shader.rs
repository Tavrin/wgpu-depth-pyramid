use crate::Reduction;

fn header(depth: bool, reduction: Reduction, count: u32) -> String {
    let mut s = format!(
        r#"
struct Params {{ src_size: vec2<u32>, offset: vec2<u32>, first_size: vec2<u32>, pitch4: u32, offset5: u32 }}
@group(0) @binding(0) var source: {texture};
@group(0) @binding(1) var<uniform> params: Params;
fn combine(a: u32, b: u32) -> u32 {{ return {op}(a,b); }}
fn load(p: vec2<u32>) -> u32 {{ return bitcast<u32>(textureLoad(source, vec2<i32>(p + params.offset), 0){component}); }}
fn end_cell(p: vec2<u32>, src: vec2<u32>, dst: vec2<u32>) -> vec2<u32> {{
    return select(min(2u*p+2u, src), src, p+1u == dst);
}}
fn reduce_source(p: vec2<u32>) -> u32 {{
    let a = min(p*2u, params.src_size-1u);
    let b = min(a+1u, params.src_size-1u);
    var v = combine(combine(load(a),load(vec2<u32>(b.x,a.y))), combine(load(vec2<u32>(a.x,b.y)),load(b)));
    let end = end_cell(p, params.src_size, params.first_size);
    if end.x > a.x+2u {{
        for (var y=a.y; y<end.y; y++) {{ v=combine(v,load(vec2<u32>(a.x+2u,y))); }}
    }}
    if end.y > a.y+2u {{
        for (var x=a.x; x<min(a.x+2u,end.x); x++) {{ v=combine(v,load(vec2<u32>(x,a.y+2u))); }}
    }}
    return v;
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
    for i in 0..count.min(4) {
        s.push_str(&format!(
            "@group(0) @binding({}) var out{i}: texture_storage_2d<r32float,write>;\n",
            i + 2
        ));
    }
    if count > 4 {
        s.push_str("@group(0) @binding(6) var<storage,read_write> extra: array<u32>;\n");
    }
    for i in 0..count {
        s.push_str(&format!("fn store{i}(p: vec2<u32>, v: u32) {{ "));
        if i < 4 {
            s.push_str(&format!(
                "textureStore(out{i},vec2<i32>(p),vec4<f32>(bitcast<f32>(v)));"
            ));
        } else if i == 4 {
            s.push_str("extra[p.y*params.pitch4+p.x]=v;");
        } else {
            s.push_str("let size=max(params.first_size >> vec2<u32>(5u),vec2<u32>(1u)); let pitch=(size.x+63u)/64u*64u; extra[params.offset5+p.y*pitch+p.x]=v;");
        }
        s.push_str(" }\n");
    }
    s
}

pub(crate) fn scalar(depth: bool, copy: bool, reduction: Reduction) -> String {
    let mut s = header(depth, reduction, 1);
    s.push_str("@compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id: vec3<u32>) { let p=id.xy; if any(p>=params.first_size) { return; } ");
    s.push_str(if copy {
        "store0(p,load(p));"
    } else {
        "store0(p,reduce_source(p));"
    });
    s.push_str(" }");
    s
}

pub(crate) fn merged(depth: bool, reduction: Reduction) -> String {
    let mut s = header(depth, reduction, 2);
    s.push_str(
        r#"
@compute @workgroup_size(16,16) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size=max(params.first_size/2u,vec2<u32>(1u)); let p=id.xy;
    if any(p>=size) { return; }
    let begin=min(p*2u,params.first_size-1u); let end=end_cell(p,params.first_size,size);
    // Keep the usual 2x2 footprint in registers with fixed load/store sites.
    // Only unmatched odd edges use loops. Clamp duplicated loads on thin axes,
    // but write each mip-zero texel exactly once.
    let b=min(begin+1u,params.first_size-1u);
    let v00=load(begin); let v10=load(vec2<u32>(b.x,begin.y));
    let v01=load(vec2<u32>(begin.x,b.y)); let v11=load(b);
    store0(begin,v00);
    if b.x!=begin.x { store0(vec2<u32>(b.x,begin.y),v10); }
    if b.y!=begin.y { store0(vec2<u32>(begin.x,b.y),v01); }
    if all(b!=begin) { store0(b,v11); }
    var v=combine(combine(v00,v10),combine(v01,v11));
    if end.x>begin.x+2u {
        for(var y=begin.y; y<end.y; y++) {
            let q=vec2<u32>(begin.x+2u,y); let a=load(q); store0(q,a); v=combine(v,a);
        }
    }
    if end.y>begin.y+2u {
        for(var x=begin.x; x<min(begin.x+2u,end.x); x++) {
            let q=vec2<u32>(x,begin.y+2u); let a=load(q); store0(q,a); v=combine(v,a);
        }
    }
    store1(p,v);
}
"#,
    );
    s
}

pub(crate) fn fused(reduction: Reduction, count: u32, tile: u32) -> String {
    // A 32-source tile reduces 2x2 per invocation. A 64-source tile reduces
    // four such quads then their parent in registers, bypassing one barrier.
    let register_stage = u32::from(tile == 64 && count > 1);
    let width = if register_stage == 1 {
        32
    } else if tile == 64 {
        64
    } else {
        32
    };
    // count=1 has no shared memory. Other paths share at most 31x31 values.
    let mut s = header(false, reduction, count);
    if count > register_stage + 1 {
        s.push_str("var<workgroup> scratch: array<u32,1280>;\n");
    }
    s.push_str(&format!(r#"
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {{
    let tile={first_tile}u;
    let groups=max(params.first_size/tile,vec2<u32>(1u));
    let base=group.xy*tile;
    let extent=select(vec2<u32>(tile),params.first_size-base,group.xy+1u==groups);
    let reg_size=max(extent >> vec2<u32>({register_stage}u),vec2<u32>(1u));
    for(var i=lane; i<reg_size.x*reg_size.y; i+=256u) {{
        let p=vec2<u32>(i%reg_size.x,i/reg_size.x);
"#,first_tile=tile/2));
    if register_stage == 0 {
        s.push_str("let value=reduce_source(base+p); store0(base+p,value);\n");
    } else {
        s.push_str(r#"
        let begin=min(p*2u,extent-1u); let end=end_cell(p,extent,reg_size);
        var value=IDENTITYu;
        for(var y=begin.y; y<end.y; y++) { for(var x=begin.x; x<end.x; x++) {
            let q=vec2<u32>(x,y); let v=reduce_source(base+q); store0(base+q,v); value=combine(value,v);
        }}
        store1((base/2u)+p,value);
"#);
    }
    if count > register_stage + 1 {
        s.push_str(&format!("scratch[p.y*{width}u+p.x]=value;\n"));
    }
    s.push_str("}\n");
    for stage in register_stage + 1..count {
        let previous = if (stage - register_stage) % 2 == 1 {
            0
        } else {
            1024
        };
        let next = 1024 - previous;
        let prev_width = if previous == 0 { 32 } else { 16 };
        let next_width = if next == 0 { 32 } else { 16 };
        s.push_str(&format!(
            r#"
    workgroupBarrier();
    let prev{stage}=max(extent >> vec2<u32>({prev_stage}u),vec2<u32>(1u));
    let size{stage}=max(extent >> vec2<u32>({stage}u),vec2<u32>(1u));
    for(var i=lane; i<size{stage}.x*size{stage}.y; i+=256u) {{
        let p=vec2<u32>(i%size{stage}.x,i/size{stage}.x);
        let begin=min(p*2u,prev{stage}-1u); let end=end_cell(p,prev{stage},size{stage});
        var value=scratch[{previous}u+begin.y*{prev_width}u+begin.x];
        for(var y=begin.y; y<end.y; y++) {{ for(var x=begin.x; x<end.x; x++) {{
            value=combine(value,scratch[{previous}u+y*{prev_width}u+x]);
        }}}}
        store{stage}((base >> vec2<u32>({stage}u))+p,value);
"#,
            prev_stage = stage - 1
        ));
        if stage + 1 < count {
            s.push_str(&format!("scratch[{next}u+p.y*{next_width}u+p.x]=value;\n"));
        }
        s.push_str("}\n");
    }
    s.push_str("}\n");
    s.replace(
        "IDENTITY",
        if reduction == Reduction::Min {
            "4294967295"
        } else {
            "0"
        },
    )
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
    fn shaders_validate() {
        for reduction in [Reduction::Min, Reduction::Max] {
            for depth in [false, true] {
                for copy in [false, true] {
                    validate(&scalar(depth, copy, reduction));
                }
                validate(&merged(depth, reduction));
            }
            for tile in [32, 64] {
                for count in 1..=if tile == 32 { 5 } else { 6 } {
                    validate(&fused(reduction, count, tile));
                }
            }
        }
    }
}
