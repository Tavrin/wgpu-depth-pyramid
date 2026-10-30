use crate::support;
use std::{path::Path, time::Instant};
use wgpu::util::DeviceExt;
use wgpu_depth_pyramid::{mip_sizes, reference, PyramidKernel, Reduction, Strategy, Viewport};
const FRAMES: u32 = 40;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.
}
fn load(device: &wgpu::Device, path: &Path, depth: bool) -> wgpu::ComputePipeline {
    let text = std::fs::read_to_string(path).expect("reference WGSL file");
    let parsed = wgpu::naga::front::wgsl::parse_str(&text)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::empty(),
    )
    .validate(&parsed)
    .unwrap();
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: if depth {
                        wgpu::TextureSampleType::Depth
                    } else {
                        wgpu::TextureSampleType::Float { filterable: false }
                    },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::R32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("reference WGSL"),
        source: wgpu::ShaderSource::Wgsl(text.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pl),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}
struct Step {
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
    groups: [u32; 2],
}
pub struct Chain {
    texture: wgpu::Texture,
    sizes: Vec<[u32; 2]>,
    steps: Vec<Step>,
}
impl Chain {
    fn new(
        device: &wgpu::Device,
        source: &wgpu::Texture,
        count: usize,
        copy: &wgpu::ComputePipeline,
        reduce: &wgpu::ComputePipeline,
        extra: [u32; 4],
    ) -> Self {
        let mut sizes = mip_sizes([source.width(), source.height()]).unwrap();
        sizes.truncate(count);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("reference chain"),
            size: source.size(),
            mip_level_count: sizes.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let views: Vec<_> = (0..sizes.len())
            .map(|i| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: i as u32,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let source_view = source.create_view(&Default::default());
        let mut steps = Vec::new();
        for (i, &size) in sizes.iter().enumerate() {
            let pipeline = if i == 0 { copy } else { reduce };
            let src = sizes[i.saturating_sub(1)];
            let words = if i == 0 {
                vec![size[0], size[1]]
            } else {
                vec![
                    src[0], src[1], size[0], size[1], extra[0], extra[1], extra[2], extra[3],
                ]
            };
            let bytes: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
            let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("cached immutable parameters"),
                contents: &bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cached per-mip bindings"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(if i == 0 {
                            &source_view
                        } else {
                            &views[i - 1]
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&views[i]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
            steps.push(Step {
                pipeline: pipeline.clone(),
                bindings,
                groups: size.map(|n| n.div_ceil(8)),
            });
        }
        Self {
            texture,
            sizes,
            steps,
        }
    }
    fn record(&self, encoder: &mut wgpu::CommandEncoder) {
        for step in &self.steps {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cached per-mip reference"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&step.pipeline);
            pass.set_bind_group(0, &step.bindings, &[]);
            pass.dispatch_workgroups(step.groups[0], step.groups[1], 1);
        }
    }
}
fn marker(encoder: &mut wgpu::CommandEncoder, queries: &wgpu::QuerySet, index: u32, start: bool) {
    let _pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("timestamp boundary"),
        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
            query_set: queries,
            beginning_of_pass_write_index: start.then_some(index),
            end_of_pass_write_index: (!start).then_some(index),
        }),
    });
}
fn read_queries(gpu: &support::Gpu, queries: &wgpu::QuerySet, count: u32) -> Vec<u64> {
    let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(count) * 8,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: resolve.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.resolve_query_set(queries, 0..count, &resolve, 0);
    encoder.copy_buffer_to_buffer(&resolve, 0, &read, 0, resolve.size());
    gpu.queue.submit([encoder.finish()]);
    gpu.map(&read)
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| u64::from_le_bytes(*b))
        .collect()
}
pub fn run() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: workload <reference WGSL directory>");
    let dir = Path::new(&dir);
    let gpu = support::Gpu::new(true);
    let kernel = PyramidKernel::new(&gpu.device);
    let copy = load(&gpu.device, &dir.join("copy.wgsl"), true);
    let reduce = load(&gpu.device, &dir.join("reduce.wgsl"), false);
    let weighted = load(&gpu.device, &dir.join("weighted.wgsl"), false);
    println!("adapter={:?}; source=Depth32Float; samples=1; full render resolution; min/max full chain; AO 5 mips; radius=0.5; falloff=0.615; period_ns={}",gpu.info,gpu.queue.get_timestamp_period());
    println!("arms=0:reference cached per-mip,1:crate scalar plus reference AO,2:crate fused plus reference AO; frames=40; warmups=8");
    for size in [[1920, 1080], [2560, 1440], [3840, 2160]] {
        let data = support::depths(size, 0);
        let source = gpu.source(size, &data, true);
        let viewport = Viewport {
            offset: [0, 0],
            size,
        };
        let full = mip_sizes(size).unwrap().len();
        let refs = [
            Chain::new(&gpu.device, &source, full, &copy, &reduce, [1, 0, 0, 0]),
            Chain::new(&gpu.device, &source, full, &copy, &reduce, [0; 4]),
        ];
        let scalar = [Reduction::Min, Reduction::Max].map(|r| {
            kernel
                .prepare_with_strategy(&gpu.device, &source, viewport, r, Strategy::Naive)
                .unwrap()
        });
        let strategy = Strategy::FusedMerged {
            tile: 32,
            depth: if size == [1920, 1080] { 5 } else { 4 },
        };
        let fused = [Reduction::Min, Reduction::Max]
            .map(|r| kernel.prepare(&gpu.device, &source, viewport, r).unwrap());
        assert!(fused.iter().all(|p| p.strategy() == strategy));
        let ao: Vec<_> = (0..3)
            .map(|_| {
                Chain::new(
                    &gpu.device,
                    &source,
                    5,
                    &copy,
                    &weighted,
                    [0.5f32.to_bits(), 0.615f32.to_bits(), 0, 0],
                )
            })
            .collect();
        let record_chain = |arm: usize, chain: usize, encoder: &mut wgpu::CommandEncoder| {
            if chain == 2 {
                ao[arm].record(encoder)
            } else {
                match arm {
                    0 => refs[chain].record(encoder),
                    1 => scalar[chain].record(encoder),
                    _ => fused[chain].record(encoder),
                }
            }
        };
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        for arm in 0..3 {
            for chain in 0..3 {
                record_chain(arm, chain, &mut encoder);
            }
        }
        gpu.queue.submit([encoder.finish()]);
        let source_bits = gpu.read_chain(&source, &[size]);
        assert!(source_bits[0]
            .iter()
            .zip(&data)
            .all(|(a, b)| *a == b.to_bits()));
        for (chain, reduction) in [Reduction::Min, Reduction::Max].into_iter().enumerate() {
            let expected = reference::build(&data, size, viewport, reduction).unwrap();
            for actual in [
                gpu.read_chain(&refs[chain].texture, &refs[chain].sizes),
                gpu.read_chain(scalar[chain].texture(), scalar[chain].sizes()),
                gpu.read_chain(fused[chain].texture(), fused[chain].sizes()),
            ] {
                assert_eq!(actual.len(), expected.len());
                for (a, e) in actual.iter().zip(&expected) {
                    assert_eq!(a.len(), e.texels.len());
                    for (&a, b) in a.iter().zip(&e.texels) {
                        assert_eq!(a, b.to_bits());
                    }
                }
            }
        }
        let ao_ref = gpu.read_chain(&ao[0].texture, &ao[0].sizes);
        assert_eq!(ao_ref.len(), 5);
        assert_eq!(ao_ref[0], source_bits[0]);
        for a in &ao[1..] {
            assert_eq!(gpu.read_chain(&a.texture, &a.sizes), ao_ref);
        }
        println!("configuration,{size:?},min_mips={full},max_mips={full},ao_mips=5,reference_passes={},scalar_passes={},fused_passes={},fused_staged_copies={},strategy={strategy:?}; min/max CPU bitwise and AO cross-arm bitwise PASS",full*2+5,scalar.iter().map(|p|p.dispatch_count()).sum::<usize>()+5,fused.iter().map(|p|p.dispatch_count()).sum::<usize>()+5,fused.iter().map(|p|p.staged_copy_count()).sum::<usize>());
        for frame in 0..8 {
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            for k in 0..3 {
                let arm = (frame + k) % 3;
                for chain in 0..3 {
                    record_chain(arm, chain, &mut encoder);
                }
            }
            gpu.queue.submit([encoder.finish()]);
        }
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let count = FRAMES * 3 * 8;
        let queries = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: None,
            ty: wgpu::QueryType::Timestamp,
            count,
        });
        let mut cpu: [Vec<f64>; 3] = Default::default();
        for frame in 0..FRAMES {
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            let order = if frame % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] };
            for k in 0..3 {
                let arm = order[((k + frame / 2) % 3) as usize];
                let q = (frame * 3 + arm) * 8;
                marker(&mut encoder, &queries, q, true);
                let mut host = 0.;
                for chain in 0..3 {
                    marker(&mut encoder, &queries, q + 2 + chain * 2, true);
                    let start = Instant::now();
                    record_chain(arm as usize, chain as usize, &mut encoder);
                    host += start.elapsed().as_secs_f64() * 1e6;
                    marker(&mut encoder, &queries, q + 3 + chain * 2, false);
                }
                cpu[arm as usize].push(host);
                marker(&mut encoder, &queries, q + 1, false);
            }
            gpu.queue.submit([encoder.finish()]);
        }
        let ticks = read_queries(&gpu, &queries, count);
        let scale = f64::from(gpu.queue.get_timestamp_period()) / 1000.;
        for arm in 0..3 {
            for metric in 0..4 {
                let mut times = Vec::new();
                for frame in 0..FRAMES {
                    let q = ((frame * 3 + arm) * 8 + metric * 2) as usize;
                    assert!(ticks[q + 1] > ticks[q]);
                    let us = (ticks[q + 1] - ticks[q]) as f64 * scale;
                    times.push(us);
                    eprintln!(
                        "work_sample,{size:?},{frame},{arm},{metric},{},{},{us:.6},{:.6}",
                        ticks[q],
                        ticks[q + 1],
                        cpu[arm as usize][frame as usize]
                    );
                }
                println!(
                    "work_row,{size:?},{arm},{metric},{:.3},{:.3}",
                    median(&mut times),
                    median(&mut cpu[arm as usize].clone())
                );
            }
        }
    }
}
