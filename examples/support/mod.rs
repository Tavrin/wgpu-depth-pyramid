#![allow(dead_code)]
use std::{
    future::Future,
    sync::{mpsc, Arc},
    task::{Context, Poll, Wake, Waker},
};

struct ThreadWake(std::thread::Thread);
impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}
pub fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: wgpu::AdapterInfo,
    depth_pipeline: wgpu::RenderPipeline,
}
impl Gpu {
    pub fn new(timestamps: bool) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .expect("GPU opt-in requires a working adapter");
        let info = adapter.get_info();
        assert_ne!(
            info.device_type,
            wgpu::DeviceType::Cpu,
            "hardware adapter required"
        );
        eprintln!("adapter={info:?}");
        let features = if timestamps {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };
        assert!(
            adapter.features().contains(features),
            "timestamp queries required for benchmark"
        );
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: features,
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .expect("default WebGPU limits");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("depth upload"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
@group(0) @binding(0) var input: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = array<vec2<f32>, 3>(vec2<f32>(-1., -1.), vec2<f32>(3., -1.), vec2<f32>(-1., 3.));
    return vec4<f32>(p[i], 0., 1.);
}
@fragment fn fs(@builtin(position) p: vec4<f32>) -> @builtin(frag_depth) f32 {
    let value = textureLoad(input, vec2<i32>(p.xy), 0).r;
    let bits = bitcast<u32>(value);
    // Preserve the cleared subnormal; fragment depth clamping may flush it.
    if bits > 0u && bits < 0x00800000u { discard; }
    return value;
}
"#
                .into(),
            ),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("depth upload"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[],
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            device,
            queue,
            info,
            depth_pipeline,
        }
    }

    pub fn source(&self, size: [u32; 2], data: &[f32], depth: bool) -> wgpu::Texture {
        let extent = wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("float source"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let bytes: Vec<_> = data
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        self.queue.write_texture(
            texture.as_image_copy(),
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * size[0]),
                rows_per_image: None,
            },
            extent,
        );
        if !depth {
            return texture;
        }
        // Depth buffer uploads are forbidden by WebGPU. A clear initializes the
        // one subnormal encoding used by this fixture, then normal fragments
        // overwrite other pixels. Discard preserves the cleared subnormal bits.
        let subnormal = data
            .iter()
            .map(|v| v.to_bits())
            .find(|b| (1..0x00800000).contains(b))
            .unwrap_or(0);
        assert!(
            data.iter().all(|v| {
                let b = v.to_bits();
                b == 0 || b >= 0x00800000 || b == subnormal
            }),
            "depth fixture uses one subnormal encoding per texture"
        );
        let output = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth source"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.depth_pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &texture.create_view(&Default::default()),
                ),
            }],
        });
        let view = output.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("depth upload"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(f32::from_bits(subnormal)),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.depth_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
        output
    }

    pub fn map(&self, buffer: &wgpu::Buffer) -> Vec<u8> {
        let (tx, rx) = mpsc::channel();
        buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            tx.send(result).unwrap()
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        rx.recv().unwrap().unwrap();
        let bytes = buffer.get_mapped_range(..).unwrap().to_vec();
        buffer.unmap();
        bytes
    }

    pub fn read_chain(&self, texture: &wgpu::Texture, sizes: &[[u32; 2]]) -> Vec<Vec<u32>> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let mut offsets = Vec::new();
        let mut total = 0;
        for size in sizes {
            let stride = (size[0] * 4).div_ceil(256) * 256;
            offsets.push((total, stride));
            total += u64::from(stride) * u64::from(size[1]);
        }
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mip readback"),
            size: total,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        for (level, (size, &(offset, stride))) in sizes.iter().zip(&offsets).enumerate() {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: if texture.format() == wgpu::TextureFormat::Depth32Float {
                        wgpu::TextureAspect::DepthOnly
                    } else {
                        wgpu::TextureAspect::All
                    },
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset,
                        bytes_per_row: Some(stride),
                        rows_per_image: None,
                    },
                },
                wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
            );
        }
        self.queue.submit([encoder.finish()]);
        let bytes = self.map(&buffer);
        sizes
            .iter()
            .zip(offsets)
            .map(|(size, (offset, stride))| {
                let mut values = Vec::with_capacity(size[0] as usize * size[1] as usize);
                for y in 0..size[1] {
                    let start = (offset + u64::from(y) * u64::from(stride)) as usize;
                    values.extend(
                        bytes[start..start + size[0] as usize * 4]
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|b| u32::from_le_bytes(*b)),
                    );
                }
                values
            })
            .collect()
    }
}

pub fn depths(size: [u32; 2], pattern: u32) -> Vec<f32> {
    let mut state = 0x72f4e51du32;
    (0..size[0] as usize * size[1] as usize)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            match pattern {
                0 => (state & 0x00ffffff) as f32 / 16777216.0,
                1 => 0.0,
                2 => 1.0,
                3 => 0.375,
                4 => f32::from_bits(1 + state % 0x007fffff),
                5 => [
                    0.0,
                    1.0,
                    f32::from_bits(1),
                    f32::from_bits(0x007fffff),
                    0.5,
                    0.5,
                ][i % 6],
                6 => {
                    if i % size[0] as usize + 1 == size[0] as usize
                        || i / size[0] as usize + 1 == size[1] as usize
                    {
                        1.0
                    } else {
                        0.25
                    }
                }
                7 => {
                    if i % size[0] as usize + 1 == size[0] as usize
                        || i / size[0] as usize + 1 == size[1] as usize
                    {
                        0.0
                    } else {
                        0.75
                    }
                }
                8 => f32::from_bits(1),
                _ => f32::from_bits(0x007fffff),
            }
        })
        .collect()
}
