//! CPU encoding comparison with a preallocated render-like buffer pool.
#![forbid(unsafe_code)]
mod support;
use std::time::Instant;
use wgpu_depth_pyramid::{Pyramid, PyramidKernel, Reduction, Strategy, Viewport};

const FRAMES: usize = 200;
const ACTIVE_BUFFERS: usize = 512;

struct Prelude {
    _buffers: Vec<wgpu::Buffer>,
    pipeline: wgpu::ComputePipeline,
    groups: Vec<wgpu::BindGroup>,
}
impl Prelude {
    fn new(device: &wgpu::Device, live: usize) -> Self {
        let buffers: Vec<_> = (0..live)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("live object or mesh buffer"),
                    size: 256,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prelude output"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let entries: Vec<_> = (0..5)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: binding != 4,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("preallocated frame bindings"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("frame prelude"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
@group(0) @binding(0) var<storage,read> a: array<u32>;
@group(0) @binding(1) var<storage,read> b: array<u32>;
@group(0) @binding(2) var<storage,read> c: array<u32>;
@group(0) @binding(3) var<storage,read> d: array<u32>;
@group(0) @binding(4) var<storage,read_write> output: array<u32>;
@compute @workgroup_size(1) fn main() { output[0]=a[0]+b[0]+c[0]+d[0]; }
"#
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let groups = buffers[..ACTIVE_BUFFERS]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| {
                let mut bindings: Vec<_> = chunk
                    .iter()
                    .zip(0..4)
                    .map(|(buffer, binding)| wgpu::BindGroupEntry {
                        binding,
                        resource: buffer.as_entire_binding(),
                    })
                    .collect();
                bindings.push(wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output.as_entire_binding(),
                });
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &layout,
                    entries: &bindings,
                })
            })
            .collect();
        Self {
            _buffers: buffers,
            pipeline,
            groups,
        }
    }
    fn record(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("frame prelude"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        for group in &self.groups {
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }
}
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
}
fn encode(
    gpu: &support::Gpu,
    prelude: &Prelude,
    pyramids: &[Pyramid],
    separate: bool,
) -> (wgpu::CommandBuffer, f64, f64) {
    let begin = Instant::now();
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    prelude.record(&mut encoder);
    let record_begin = Instant::now();
    for pyramid in pyramids {
        if separate {
            pyramid.record_separate_passes(&mut encoder);
        } else {
            pyramid.record(&mut encoder);
        }
    }
    let record_us = record_begin.elapsed().as_secs_f64() * 1e6;
    let command = encoder.finish();
    (command, record_us, begin.elapsed().as_secs_f64() * 1e6)
}
fn main() {
    let gpu = support::Gpu::new(false);
    let kernel = PyramidKernel::new(&gpu.device);
    let size = [1920, 1080];
    let source = gpu.source(size, &support::depths(size, 0), true);
    println!("adapter={:?}; default limits; optional features=empty; 1080p Depth32Float; full Min+Max+AO chains per frame; frames={FRAMES}; warmup_pairs=20; prelude_passes=1; prelude_dispatches=128",gpu.info);
    println!("pool_buffers,active_pool_buffers,strategy,dispatches,one_pyramid_passes,separate_pyramid_passes,one_record_us,separate_record_us,one_full_encode_us,separate_full_encode_us,one_submit_us,separate_submit_us");
    for live in [1024, 4096, 16384] {
        let prelude = Prelude::new(&gpu.device, live);
        for strategy in [Strategy::Auto, Strategy::Naive] {
            let pyramids: Vec<_> = [
                Reduction::Min,
                Reduction::Max,
                Reduction::Ao {
                    effect_radius: 0.5,
                    effect_falloff_range: 0.615,
                },
            ]
            .into_iter()
            .map(|r| {
                kernel
                    .prepare_with_strategy(
                        &gpu.device,
                        &source,
                        Viewport::new([0, 0], size),
                        r,
                        strategy,
                    )
                    .unwrap()
            })
            .collect();
            assert!(pyramids
                .iter()
                .all(|p| p.compute_pass_count() == 1 && p.staged_copy_count() == 0));
            // Validate every mip of all three complete chains in both recording modes.
            let (command, _, _) = encode(&gpu, &prelude, &pyramids, false);
            gpu.queue.submit([command]);
            let one: Vec<_> = pyramids
                .iter()
                .map(|p| gpu.read_chain(p.texture(), p.sizes()))
                .collect();
            let (command, _, _) = encode(&gpu, &prelude, &pyramids, true);
            gpu.queue.submit([command]);
            for (p, expected) in pyramids.iter().zip(one) {
                assert_eq!(
                    gpu.read_chain(p.texture(), p.sizes()),
                    expected,
                    "{strategy:?} one-pass vs separate-pass identity"
                );
            }
            for frame in 0..20 {
                for separate in [frame % 2 == 0, frame % 2 != 0] {
                    let (command, _, _) = encode(&gpu, &prelude, &pyramids, separate);
                    gpu.queue.submit([command]);
                }
            }
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            let mut record: [Vec<f64>; 2] = Default::default();
            let mut full: [Vec<f64>; 2] = Default::default();
            let mut submit: [Vec<f64>; 2] = Default::default();
            for frame in 0..FRAMES {
                for arm in [frame % 2, 1 - frame % 2] {
                    let (command, record_us, full_us) = encode(&gpu, &prelude, &pyramids, arm == 1);
                    let submit_begin = Instant::now();
                    gpu.queue.submit([command]);
                    let submit_us = submit_begin.elapsed().as_secs_f64() * 1e6;
                    record[arm].push(record_us);
                    full[arm].push(full_us);
                    submit[arm].push(submit_us);
                    eprintln!("encode_sample,{live},{strategy:?},{frame},{arm},{record_us:.6},{full_us:.6},{submit_us:.6}");
                }
                gpu.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .unwrap();
            }
            let record = record.each_mut().map(|v| median(v));
            let full = full.each_mut().map(|v| median(v));
            let submit = submit.each_mut().map(|v| median(v));
            let dispatches: usize = pyramids.iter().map(Pyramid::dispatch_count).sum();
            println!("encode,{live},{ACTIVE_BUFFERS},{strategy:?},{dispatches},3,{dispatches},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",record[0],record[1],full[0],full[1],submit[0],submit[1]);
        }
    }
}
