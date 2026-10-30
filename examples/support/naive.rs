use wgpu::util::DeviceExt;
use wgpu_depth_pyramid::{mip_sizes, Reduction, Viewport};

// A deliberately conventional baseline: a distinct compute pass and 8x8 dispatch
// per mip, with all uniforms and bindings prepared before timing (a strong baseline).
pub struct Naive {
    pub texture: wgpu::Texture,
    pub sizes: Vec<[u32; 2]>,
    steps: Vec<(wgpu::ComputePipeline, wgpu::BindGroup, [u32; 2])>,
}
impl Naive {
    #[allow(dead_code)]
    pub fn new(device: &wgpu::Device, source: &wgpu::Texture, reduction: Reduction) -> Self {
        Self::with_viewport(
            device,
            source,
            Viewport::new([0, 0], [source.width(), source.height()]),
            reduction,
        )
    }
    pub fn with_viewport(
        device: &wgpu::Device,
        source: &wgpu::Texture,
        viewport: Viewport,
        reduction: Reduction,
    ) -> Self {
        let sizes = mip_sizes(viewport.size).unwrap();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("naive pyramid"),
            size: wgpu::Extent3d {
                width: viewport.size[0],
                height: viewport.size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: sizes.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let layouts: Vec<_> = [true, false]
            .into_iter()
            .map(|copy| {
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: if copy
                                    && source.format() == wgpu::TextureFormat::Depth32Float
                                {
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
                                min_binding_size: std::num::NonZeroU64::new(32),
                            },
                            count: None,
                        },
                    ],
                })
            })
            .collect();
        let pipelines: Vec<_> = [true, false]
            .into_iter()
            .enumerate()
            .map(|(i, copy)| {
                let pipeline_layout =
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[Some(&layouts[i])],
                        immediate_size: 0,
                    });
                let mut code = include_str!("naive.wgsl")
                    .replace("COPY_BASE", if copy { "true" } else { "false" })
                    .replace(
                        "REDUCE",
                        if reduction == Reduction::Min {
                            "min"
                        } else {
                            "max"
                        },
                    );
                if let Reduction::Ao { .. } = reduction {
                    code = include_str!("ao_reference.wgsl")
                        .replace("COPY_BASE", if copy { "true" } else { "false" })
                        .replace("COMPONENT", ".r");
                }
                if copy && source.format() == wgpu::TextureFormat::Depth32Float {
                    code = code
                        .replace("texture_2d<f32>", "texture_depth_2d")
                        .replace("0).r", "0)");
                }
                let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("naive WGSL"),
                    source: wgpu::ShaderSource::Wgsl(code.into()),
                });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("naive"),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();
        let views: Vec<_> = (0..sizes.len())
            .map(|level| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: level as u32,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let source_view = source.create_view(&wgpu::TextureViewDescriptor {
            mip_level_count: Some(1),
            array_layer_count: Some(1),
            ..Default::default()
        });
        let steps = sizes
            .iter()
            .enumerate()
            .map(|(i, size)| {
                let src_size = sizes[i.saturating_sub(1)];
                let (radius, falloff) = match reduction {
                    Reduction::Ao {
                        effect_radius,
                        effect_falloff_range,
                    } => (effect_radius, effect_falloff_range),
                    _ => (0.0f32, 0.0f32),
                };
                let bytes: Vec<_> = [
                    src_size[0],
                    src_size[1],
                    size[0],
                    size[1],
                    radius.to_bits(),
                    falloff.to_bits(),
                    viewport.offset[0],
                    viewport.offset[1],
                ]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect();
                let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: &bytes,
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &layouts[usize::from(i != 0)],
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
                (
                    pipelines[usize::from(i != 0)].clone(),
                    bind,
                    size.map(|n| n.div_ceil(8)),
                )
            })
            .collect();
        Self {
            texture,
            sizes,
            steps,
        }
    }
    pub fn record(&self, encoder: &mut wgpu::CommandEncoder) {
        for (pipeline, bind, groups) in &self.steps {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("naive mip"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(groups[0], groups[1], 1);
        }
    }
}
