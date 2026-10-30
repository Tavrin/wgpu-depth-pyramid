#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![doc = include_str!("../README.md")]

use std::{fmt, num::NonZeroU64};
use wgpu::util::DeviceExt;

mod ao;
mod legacy_shader;
pub mod reference;
mod shader;

/// Reduction is explicit: a reverse-Z depth has larger values nearer the camera.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reduction {
    /// Farthest surface in reverse-Z; appropriate for conservative occlusion tests.
    Min,
    /// Closest surface in reverse-Z; appropriate for hierarchical ray marching.
    Max,
    /// Reverse-Z depth-aware weighted downsample for screen-space ambient occlusion.
    /// Uses the f32 formula and clamped 2x2 edge rule in the crate documentation.
    /// GPU identity is bit-exact against the ported reference shader on the same
    /// adapter. CPU comparisons use the propagated 2.5-ULP-per-division bound
    /// documented in [`reference::build`].
    Ao {
        /// Finite nonnegative radius in the same units as the source depth.
        effect_radius: f32,
        /// Fraction of the radius over which weights fall off, in [0, 1].
        effect_falloff_range: f32,
    },
}

impl Reduction {
    pub(crate) const AO: Self = Self::Ao {
        effect_radius: 0.5,
        effect_falloff_range: 0.615,
    };

    pub(crate) fn is_ao(self) -> bool {
        matches!(self, Self::Ao { .. })
    }

    fn pipeline_key(self) -> Self {
        if self.is_ao() {
            Self::AO
        } else {
            self
        }
    }

    pub(crate) fn validate(self) -> Result<(), Error> {
        if let Self::Ao {
            effect_radius,
            effect_falloff_range,
        } = self
        {
            let radius = 0.75 * effect_radius;
            let range = (effect_falloff_range * radius).max(1e-5);
            let add = radius * (1.0 - effect_falloff_range) / range + 1.0;
            if !effect_radius.is_finite()
                || effect_radius < 0.0
                || !(0.0..=1.0).contains(&effect_falloff_range)
                || !add.is_finite()
            {
                return Err(Error("AO requires a finite nonnegative radius, falloff in [0,1], and finite derived coefficients"));
            }
        }
        Ok(())
    }
}

/// Source rectangle, in texels. Output mip zero is an exact copy of this rectangle.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Viewport {
    /// Origin of the source rectangle, in texels.
    pub offset: [u32; 2],
    /// Nonzero width and height of the source rectangle.
    pub size: [u32; 2],
}

impl Viewport {
    /// Construct a source rectangle. Bounds are checked when preparing a pyramid.
    pub const fn new(offset: [u32; 2], size: [u32; 2]) -> Self {
        Self { offset, size }
    }
}

/// Invalid setup. GPU validation still applies to device ownership and texture usages.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error(
    /// Human-readable setup failure.
    pub &'static str,
);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}
impl std::error::Error for Error {}

pub(crate) fn validate(size: [u32; 2], viewport: Viewport) -> Result<(), Error> {
    for (axis, bound) in size.into_iter().enumerate() {
        if viewport.size[axis] == 0
            || viewport.offset[axis]
                .checked_add(viewport.size[axis])
                .is_none_or(|end| end > bound)
        {
            return Err(Error(
                "viewport must be nonempty and contained in the source",
            ));
        }
    }
    Ok(())
}

/// Dimensions of a full chain, including level zero; floor-halved, clamped to one.
pub fn mip_sizes(size: [u32; 2]) -> Result<Vec<[u32; 2]>, Error> {
    validate(
        size,
        Viewport {
            offset: [0, 0],
            size,
        },
    )?;
    let mut result = vec![size];
    let mut size = size;
    while size != [1, 1] {
        size = size.map(|n| (n / 2).max(1));
        result.push(size);
    }
    Ok(result)
}

/// Explicit implementation choice. `Auto` uses the measured resolution policy.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    /// Use fused paths at 1080p, 1440p and 4K; scalar elsewhere.
    Auto,
    /// Conventional 8x8, one dispatch per mip, persistent parameters/bindings.
    Naive,
    /// Copy and first reduction in registers, then conventional per-mip dispatches.
    Merged,
    /// Tile-local reductions; tile is a source tile (32 or 64), depth is 2..=6.
    /// A 32-wide tile supports at most five reductions.
    Fused {
        /// Source tile width and height: 32 or 64 texels.
        tile: u32,
        /// Maximum reductions per tile: 2..=5 for 32, 2..=6 for 64.
        depth: u32,
    },
    /// Merge mip-zero copy with the first reduction, then fuse the remaining levels.
    FusedMerged {
        /// Source tile width and height: 32 or 64 texels.
        tile: u32,
        /// Maximum reductions per tile: 2..=5 for 32, 2..=6 for 64.
        depth: u32,
    },
    /// Original 64-thread shared-memory implementation, for comparison.
    Legacy,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Scalar,
    Merged,
    Fused32,
    Fused64,
    Legacy,
}

struct Variant {
    mode: Mode,
    depth: bool,
    copy: bool,
    reduction: Reduction,
    count: u32,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

/// Pipelines compiled once for both source formats and all reductions.
/// Requires only WebGPU default limits and no optional features.
pub struct PyramidKernel {
    variants: Vec<Variant>,
}

impl PyramidKernel {
    /// Compile all pipeline variants for reuse across sources and resizes.
    pub fn new(device: &wgpu::Device) -> Self {
        let mut variants = Vec::new();
        for mode in [
            Mode::Scalar,
            Mode::Merged,
            Mode::Fused32,
            Mode::Fused64,
            Mode::Legacy,
        ] {
            for (depth, copy) in [(false, true), (true, true), (false, false)] {
                if matches!(mode, Mode::Fused32 | Mode::Fused64) && copy {
                    continue;
                }
                if mode == Mode::Merged && !copy {
                    continue;
                }
                for reduction in [Reduction::Min, Reduction::Max, Reduction::AO] {
                    let max_count = match mode {
                        Mode::Scalar => 1,
                        Mode::Merged => 2,
                        Mode::Legacy => 4,
                        Mode::Fused32 => 5,
                        Mode::Fused64 => 6,
                    };
                    let min_count = if mode == Mode::Merged { 2 } else { 1 };
                    for count in min_count..=max_count {
                        let mut entries = vec![
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
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: NonZeroU64::new(48),
                                },
                                count: None,
                            },
                        ];
                        for i in 0..count.min(4) {
                            entries.push(wgpu::BindGroupLayoutEntry {
                                binding: 2 + i,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::StorageTexture {
                                    access: wgpu::StorageTextureAccess::WriteOnly,
                                    format: wgpu::TextureFormat::R32Float,
                                    view_dimension: wgpu::TextureViewDimension::D2,
                                },
                                count: None,
                            });
                        }
                        if count > 4 {
                            entries.push(wgpu::BindGroupLayoutEntry {
                                binding: 6,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            });
                        }
                        let layout =
                            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                                label: Some("pyramid bindings"),
                                entries: &entries,
                            });
                        let pipeline_layout =
                            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                                label: Some("pyramid layout"),
                                bind_group_layouts: &[Some(&layout)],
                                immediate_size: 0,
                            });
                        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                            label: Some("pyramid WGSL"),
                            source: wgpu::ShaderSource::Wgsl(
                                match mode {
                                    Mode::Legacy => {
                                        legacy_shader::generate(depth, copy, reduction, count)
                                    }
                                    Mode::Scalar => shader::scalar(depth, copy, reduction),
                                    Mode::Merged => shader::merged(depth, reduction),
                                    Mode::Fused32 => shader::fused(reduction, count, 32),
                                    Mode::Fused64 => shader::fused(reduction, count, 64),
                                }
                                .into(),
                            ),
                        });
                        let pipeline =
                            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                                label: Some("pyramid fused"),
                                layout: Some(&pipeline_layout),
                                module: &module,
                                entry_point: Some("main"),
                                compilation_options: Default::default(),
                                cache: None,
                            });
                        variants.push(Variant {
                            mode,
                            depth,
                            copy,
                            reduction,
                            count,
                            layout,
                            pipeline,
                        });
                    }
                }
            }
        }
        Self { variants }
    }

    /// Allocate output, views, bind groups and immutable parameter buffers at setup/resize.
    /// Source must be a single-sampled 2D R32Float or Depth32Float texture with
    /// TEXTURE_BINDING usage. Its base mip and first array layer are read.
    /// Recreate this object whenever the source, viewport, or reduction changes.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        source: &wgpu::Texture,
        viewport: Viewport,
        reduction: Reduction,
    ) -> Result<Pyramid, Error> {
        self.prepare_with_strategy(device, source, viewport, reduction, Strategy::Auto)
    }

    /// Prepare an explicit strategy for qualification or adapter-specific tuning.
    pub fn prepare_with_strategy(
        &self,
        device: &wgpu::Device,
        source: &wgpu::Texture,
        viewport: Viewport,
        reduction: Reduction,
        strategy: Strategy,
    ) -> Result<Pyramid, Error> {
        reduction.validate()?;
        let strategy = match strategy {
            Strategy::Auto => match viewport.size {
                [1920, 1080] | [2560, 1440] | [3840, 2160] => {
                    Strategy::FusedMerged { tile: 32, depth: 4 }
                }
                _ => Strategy::Naive,
            },
            other => other,
        };
        if let Strategy::Fused { tile, depth } | Strategy::FusedMerged { tile, depth } = strategy {
            if !matches!(tile, 32 | 64) || !(2..=6).contains(&depth) || (tile == 32 && depth > 5) {
                return Err(Error(
                    "fused tile must be 32 (depth 2..=5) or 64 (depth 2..=6)",
                ));
            }
        }
        validate([source.width(), source.height()], viewport)?;
        let depth = match source.format() {
            wgpu::TextureFormat::R32Float => false,
            wgpu::TextureFormat::Depth32Float => true,
            _ => return Err(Error("source format must be R32Float or Depth32Float")),
        };
        if source.dimension() != wgpu::TextureDimension::D2
            || source.sample_count() != 1
            || !source
                .usage()
                .contains(wgpu::TextureUsages::TEXTURE_BINDING)
        {
            return Err(Error(
                "source must be single-sampled 2D with TEXTURE_BINDING usage",
            ));
        }
        let sizes = mip_sizes(viewport.size)?;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth pyramid"),
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
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
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
            dimension: Some(wgpu::TextureViewDimension::D2),
            mip_level_count: Some(1),
            array_layer_count: Some(1),
            aspect: if depth {
                wgpu::TextureAspect::DepthOnly
            } else {
                wgpu::TextureAspect::All
            },
            ..Default::default()
        });
        let mut steps = Vec::new();
        let mut start = 0;
        while start < sizes.len() {
            let copy = start == 0;
            let (mode, batch) = match strategy {
                Strategy::Auto => unreachable!(),
                Strategy::Naive => (Mode::Scalar, 1),
                Strategy::Merged => {
                    if copy && sizes.len() > 1 {
                        (Mode::Merged, 2)
                    } else {
                        (Mode::Scalar, 1)
                    }
                }
                Strategy::Legacy => (Mode::Legacy, 4),
                Strategy::FusedMerged { .. } if copy && sizes.len() > 1 => (Mode::Merged, 2),
                Strategy::Fused { tile, depth } | Strategy::FusedMerged { tile, depth } => {
                    if copy {
                        (Mode::Scalar, 1)
                    } else {
                        (
                            if tile == 32 {
                                Mode::Fused32
                            } else {
                                Mode::Fused64
                            },
                            depth as usize,
                        )
                    }
                }
            };
            let count = (sizes.len() - start).min(batch) as u32;
            let variant = self
                .variants
                .iter()
                .find(|v| {
                    v.mode == mode
                        && v.depth == (depth && copy)
                        && v.copy == copy
                        && v.reduction == reduction.pipeline_key()
                        && v.count == count
                })
                .unwrap();
            let src_size = if copy {
                viewport.size
            } else {
                sizes[start - 1]
            };
            let offset = if copy { viewport.offset } else { [0, 0] };
            let spill4 = sizes.get(start + 4).copied().unwrap_or([1, 1]);
            let spill5 = sizes.get(start + 5).copied().unwrap_or([1, 1]);
            let pitch4 = spill4[0].div_ceil(64) * 64;
            let pitch5 = spill5[0].div_ceil(64) * 64;
            let offset5 = pitch4 * spill4[1];
            let spill = (count > 4).then(|| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("portable extra mip staging"),
                    size: u64::from(offset5 + pitch5 * spill5[1]) * 4,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            });
            let (radius, falloff) = match reduction {
                Reduction::Ao {
                    effect_radius,
                    effect_falloff_range,
                } => (effect_radius, effect_falloff_range),
                _ => (0.0f32, 0.0f32),
            };
            let words = [
                src_size[0],
                src_size[1],
                offset[0],
                offset[1],
                sizes[start][0],
                sizes[start][1],
                pitch4,
                offset5,
                radius.to_bits(),
                falloff.to_bits(),
                0,
                0,
            ];
            let bytes: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pyramid immutable parameters"),
                contents: &bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(if copy {
                        &source_view
                    } else {
                        &views[start - 1]
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer.as_entire_binding(),
                },
            ];
            for i in 0..count.min(4) {
                entries.push(wgpu::BindGroupEntry {
                    binding: 2 + i,
                    resource: wgpu::BindingResource::TextureView(&views[start + i as usize]),
                });
            }
            if let Some(spill) = &spill {
                entries.push(wgpu::BindGroupEntry {
                    binding: 6,
                    resource: spill.as_entire_binding(),
                });
            }
            let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("pyramid bindings"),
                layout: &variant.layout,
                entries: &entries,
            });
            let groups = match mode {
                Mode::Legacy => sizes[start].map(|n| (n / (1 << (count - 1))).max(1)),
                Mode::Scalar => sizes[start].map(|n| n.div_ceil(8)),
                Mode::Merged => sizes[start + 1].map(|n| n.div_ceil(16)),
                Mode::Fused32 | Mode::Fused64 => {
                    let tile = if mode == Mode::Fused32 { 16 } else { 32 };
                    sizes[start].map(|n| (n / tile).max(1))
                }
            };
            steps.push(Step {
                pipeline: variant.pipeline.clone(),
                bindings,
                groups,
                spill,
                spill_levels: (4..count)
                    .map(|i| {
                        let size = sizes[start + i as usize];
                        (
                            start as u32 + i,
                            size,
                            if i == 4 { pitch4 } else { pitch5 },
                            if i == 4 { 0 } else { u64::from(offset5) * 4 },
                        )
                    })
                    .collect(),
            });
            start += count as usize;
        }
        Ok(Pyramid {
            texture,
            views,
            sizes,
            steps,
            strategy,
        })
    }
}

struct Step {
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
    groups: [u32; 2],
    spill: Option<wgpu::Buffer>,
    spill_levels: Vec<(u32, [u32; 2], u32, u64)>,
}

/// Persistent resources for one source rectangle and reduction.
pub struct Pyramid {
    texture: wgpu::Texture,
    views: Vec<wgpu::TextureView>,
    sizes: Vec<[u32; 2]>,
    steps: Vec<Step>,
    strategy: Strategy,
}
impl Pyramid {
    /// Record the complete chain in one compute pass by default.
    /// Bindings and parameters are preallocated. Each dispatch is a separate
    /// WebGPU usage scope; wgpu inserts storage-write to sampled-read barriers.
    /// Explicit strategies with staged levels split only at buffer-to-texture
    /// copies, which cannot be encoded inside a compute pass.
    pub fn record(&self, encoder: &mut wgpu::CommandEncoder) {
        self.record_inner(encoder, None, false);
    }

    /// Record one compute pass per dispatch, for profiling and comparison.
    /// Uses the same preallocated pipelines, bind groups and output texture.
    pub fn record_separate_passes(&self, encoder: &mut wgpu::CommandEncoder) {
        self.record_inner(encoder, None, true);
    }

    /// Per-dispatch timestamps for profiling; requires TIMESTAMP_QUERY.
    /// Each dispatch uses a separate pass and two queries, beginning at `first_query`.
    pub fn record_profiled(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queries: &wgpu::QuerySet,
        first_query: u32,
    ) {
        self.record_inner(encoder, Some((queries, first_query)), true);
    }

    fn record_inner(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamps: Option<(&wgpu::QuerySet, u32)>,
        separate: bool,
    ) {
        let mut start = 0;
        while start < self.steps.len() {
            let end = if separate {
                start + 1
            } else {
                self.steps[start..]
                    .iter()
                    .position(|step| step.spill.is_some())
                    .map_or(self.steps.len(), |i| start + i + 1)
            };
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pyramid chain"),
                    timestamp_writes: timestamps.map(|(q, first)| {
                        wgpu::ComputePassTimestampWrites {
                            query_set: q,
                            beginning_of_pass_write_index: Some(
                                first + 2 * u32::try_from(start).expect("mip index fits u32"),
                            ),
                            end_of_pass_write_index: Some(
                                first + 2 * u32::try_from(start).expect("mip index fits u32") + 1,
                            ),
                        }
                    }),
                });
                for step in &self.steps[start..end] {
                    pass.set_pipeline(&step.pipeline);
                    pass.set_bind_group(0, &step.bindings, &[]);
                    pass.dispatch_workgroups(step.groups[0], step.groups[1], 1);
                }
            }
            let step = &self.steps[end - 1];
            if let Some(spill) = &step.spill {
                for &(mip, size, pitch, offset) in &step.spill_levels {
                    encoder.copy_buffer_to_texture(
                        wgpu::TexelCopyBufferInfo {
                            buffer: spill,
                            layout: wgpu::TexelCopyBufferLayout {
                                offset,
                                bytes_per_row: Some(pitch * 4),
                                rows_per_image: None,
                            },
                        },
                        wgpu::TexelCopyTextureInfo {
                            texture: &self.texture,
                            mip_level: mip,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        wgpu::Extent3d {
                            width: size[0],
                            height: size[1],
                            depth_or_array_layers: 1,
                        },
                    );
                }
            }
            start = end;
        }
    }

    /// Compute passes used by [`Self::record`], excluding timestamp profiling.
    /// Automatic strategies use one pass. Explicit staged strategies split at copies.
    pub fn compute_pass_count(&self) -> usize {
        1 + self.steps[..self.steps.len() - 1]
            .iter()
            .filter(|step| step.spill.is_some())
            .count()
    }

    /// Resolved implementation choice; never `Strategy::Auto`.
    pub fn strategy(&self) -> Strategy {
        self.strategy
    }
    /// Viewport-sized R32Float output with the complete mip chain.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }
    /// One single-level view per output mip, beginning at mip zero.
    pub fn mip_views(&self) -> &[wgpu::TextureView] {
        &self.views
    }
    /// Output dimensions in mip order, including the copied viewport.
    pub fn sizes(&self) -> &[[u32; 2]] {
        &self.sizes
    }
    /// Number of portable buffer-to-texture copies in a recorded chain.
    pub fn staged_copy_count(&self) -> usize {
        self.steps.iter().map(|step| step.spill_levels.len()).sum()
    }
    /// Number of compute dispatches per recording, independent of pass grouping.
    pub fn dispatch_count(&self) -> usize {
        self.steps.len()
    }
}
