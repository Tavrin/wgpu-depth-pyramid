#![cfg(not(target_arch = "wasm32"))]
mod common;
#[path = "../examples/support/mod.rs"]
mod support;
use wgpu_depth_pyramid::{reference, PyramidKernel, Reduction, Viewport};

#[test]
#[ignore = "requires a hardware GPU; opt in explicitly under an exclusive GPU lease"]
fn every_mip_bit_exact() {
    let gpu = support::Gpu::new(false);
    let kernel = PyramidKernel::new(&gpu.device);
    let cases = common::sizes();
    let mut comparisons = 0u64;
    let mut chains = 0;
    for size in cases {
        // Every size is tested both at origin and inside a larger texture.
        for offset in [[0, 0], [3, 5]] {
            let source_size = if offset == [0, 0] {
                size
            } else {
                [size[0] + offset[0] + 2, size[1] + offset[1] + 1]
            };
            let viewport = Viewport::new(offset, size);
            for pattern in 0..10 {
                let mut data = support::depths(source_size, pattern);
                if matches!(pattern, 6 | 7) {
                    // Extremal values live at the viewport's tail, not just at the
                    // enclosing texture edge. The surrounding margin is the opposite.
                    let edge = if pattern == 6 { 1.0 } else { 0.0 };
                    data.fill(if pattern == 6 { 0.25 } else { 0.75 });
                    for y in offset[1]..offset[1] + size[1] {
                        data[(y * source_size[0] + offset[0] + size[0] - 1) as usize] = edge;
                    }
                    for x in offset[0]..offset[0] + size[0] {
                        data[((offset[1] + size[1] - 1) * source_size[0] + x) as usize] = edge;
                    }
                }
                for depth in [false, true] {
                    let input: std::borrow::Cow<'_, [f32]> = if depth && matches!(pattern, 4 | 5) {
                        let mut depth_data = data.clone();
                        for value in &mut depth_data {
                            let bits = value.to_bits();
                            if (1..0x00800000).contains(&bits) {
                                *value = f32::from_bits(if pattern == 4 { 0x007fffff } else { 1 });
                            }
                        }
                        std::borrow::Cow::Owned(depth_data)
                    } else {
                        std::borrow::Cow::Borrowed(&data)
                    };
                    let source = gpu.source(source_size, &input, depth);
                    // Independently validate the source texture before testing reduction.
                    let uploaded = gpu.read_chain(&source, &[source_size]);
                    for (i, (&actual, expected)) in uploaded[0].iter().zip(input.iter()).enumerate()
                    {
                        assert_eq!(actual, expected.to_bits(), "source initialization size={size:?} offset={offset:?} pattern={pattern} depth={depth} texel={i}");
                    }
                    for reduction in [Reduction::Min, Reduction::Max] {
                        let pyramid = kernel
                            .prepare(&gpu.device, &source, viewport, reduction)
                            .unwrap();
                        assert_eq!(pyramid.compute_pass_count(), 1);
                        let mut encoder = gpu.device.create_command_encoder(&Default::default());
                        pyramid.record(&mut encoder);
                        gpu.queue.submit([encoder.finish()]);
                        let got = gpu.read_chain(pyramid.texture(), pyramid.sizes());
                        let expected =
                            reference::build(&input, source_size, viewport, reduction).unwrap();
                        for (level, (actual, expected)) in got.iter().zip(expected).enumerate() {
                            assert_eq!(actual.len(), expected.texels.len());
                            for (i, (&a, b)) in actual.iter().zip(expected.texels).enumerate() {
                                assert_eq!(a, b.to_bits(), "size={size:?} offset={offset:?} pattern={pattern} depth={depth} reduction={reduction:?} mip={level} texel={i}");
                            }
                            comparisons += actual.len() as u64;
                        }
                        chains += 1;
                    }
                }
            }
        }
        eprintln!("PASS size={size:?}, cumulative chains={chains}, texels={comparisons}");
    }
    // Repeated recording, source content updates and resource recreation at resize.
    let viewport = Viewport::new([0, 0], [17, 19]);
    let source = gpu.source(viewport.size, &[0.0; 17 * 19], false);
    let pyramid = kernel
        .prepare(&gpu.device, &source, viewport, Reduction::Max)
        .unwrap();
    for value in [0.0f32, 0.75] {
        let bytes: Vec<_> = (0..17 * 19).flat_map(|_| value.to_le_bytes()).collect();
        gpu.queue.write_texture(
            source.as_image_copy(),
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(17 * 4),
                rows_per_image: None,
            },
            source.size(),
        );
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        pyramid.record(&mut encoder);
        gpu.queue.submit([encoder.finish()]);
        assert!(gpu
            .read_chain(pyramid.texture(), pyramid.sizes())
            .iter()
            .flatten()
            .all(|&v| v == value.to_bits()));
    }
    assert!(kernel
        .prepare(
            &gpu.device,
            &source,
            Viewport::new([1, 0], [17, 19]),
            Reduction::Min
        )
        .is_err());
    eprintln!("TOTAL chains={chains}, bitwise texel comparisons={comparisons}");
}

#[test]
#[ignore = "requires a hardware GPU; opt in explicitly under an exclusive GPU lease"]
fn explicit_variants_fold_boundaries() {
    use wgpu_depth_pyramid::Strategy;
    let gpu = support::Gpu::new(false);
    let kernel = PyramidKernel::new(&gpu.device);
    let mut strategies = vec![Strategy::Naive, Strategy::Merged, Strategy::Legacy];
    for tile in [32, 64] {
        for depth in 2..=if tile == 32 { 5 } else { 6 } {
            strategies.push(Strategy::Fused { tile, depth });
            strategies.push(Strategy::FusedMerged { tile, depth });
        }
    }
    let mut chains = 0;
    let mut comparisons = 0u64;
    for size in [
        [1, 1],
        [1, 3],
        [3, 1],
        [1, 4097],
        [4097, 1],
        [3, 5],
        [31, 33],
        [32, 32],
        [33, 63],
        [63, 65],
        [64, 64],
        [65, 127],
        [127, 129],
        [129, 255],
        [255, 257],
        [257, 129],
        [513, 259],
    ] {
        for offset in [[0, 0], [3, 5]] {
            let source_size = [size[0] + offset[0], size[1] + offset[1]];
            let viewport = Viewport::new(offset, size);
            for pattern in [0, 4, 5, 6, 7] {
                let data = support::depths(source_size, pattern);
                for depth in [false, true] {
                    let input: Vec<_> = data
                        .iter()
                        .map(|&v| {
                            if depth && v.is_subnormal() {
                                f32::from_bits(1)
                            } else {
                                v
                            }
                        })
                        .collect();
                    let source = gpu.source(source_size, &input, depth);
                    let uploaded = gpu.read_chain(&source, &[source_size]);
                    assert!(uploaded[0]
                        .iter()
                        .zip(&input)
                        .all(|(a, b)| *a == b.to_bits()));
                    for reduction in [Reduction::Min, Reduction::Max] {
                        let expected =
                            reference::build(&input, source_size, viewport, reduction).unwrap();
                        for &strategy in &strategies {
                            let pyramid = kernel
                                .prepare_with_strategy(
                                    &gpu.device,
                                    &source,
                                    viewport,
                                    reduction,
                                    strategy,
                                )
                                .unwrap();
                            let mut encoder =
                                gpu.device.create_command_encoder(&Default::default());
                            pyramid.record(&mut encoder);
                            gpu.queue.submit([encoder.finish()]);
                            let got = gpu.read_chain(pyramid.texture(), pyramid.sizes());
                            assert_eq!(got.len(), expected.len());
                            for (level, (a, e)) in got.iter().zip(&expected).enumerate() {
                                assert_eq!(a.len(), e.texels.len());
                                for (i, (&a, b)) in a.iter().zip(&e.texels).enumerate() {
                                    assert_eq!(a,b.to_bits(),"{strategy:?} {size:?} {offset:?} pattern={pattern} depth={depth} {reduction:?} mip={level} texel={i}");
                                }
                                comparisons += a.len() as u64;
                            }
                            chains += 1;
                        }
                    }
                }
            }
        }
        eprintln!("VARIANT PASS size={size:?} chains={chains} comparisons={comparisons}");
    }
    eprintln!("VARIANT TOTAL chains={chains} comparisons={comparisons}");
}

#[test]
#[ignore = "requires a hardware GPU under an exclusive GPU lease"]
fn one_pass_storage_to_sampled_dependencies_match_separate_passes() {
    use wgpu_depth_pyramid::Strategy;
    let gpu = support::Gpu::new(false);
    let kernel = PyramidKernel::new(&gpu.device);
    let size = [1920, 1080];
    let data = support::depths(size, 0);
    for depth in [false, true] {
        let source = gpu.source(size, &data, depth);
        for reduction in [
            Reduction::Min,
            Reduction::Max,
            Reduction::Ao {
                effect_radius: 0.5,
                effect_falloff_range: 0.615,
            },
        ] {
            for strategy in [Strategy::Auto, Strategy::Naive, Strategy::Legacy] {
                let pyramid = kernel
                    .prepare_with_strategy(
                        &gpu.device,
                        &source,
                        Viewport::new([0, 0], size),
                        reduction,
                        strategy,
                    )
                    .unwrap();
                assert_eq!(pyramid.compute_pass_count(), 1);
                assert_eq!(pyramid.staged_copy_count(), 0);
                let mut encoder = gpu.device.create_command_encoder(&Default::default());
                pyramid.record(&mut encoder);
                gpu.queue.submit([encoder.finish()]);
                let one = gpu.read_chain(pyramid.texture(), pyramid.sizes());
                let mut encoder = gpu.device.create_command_encoder(&Default::default());
                pyramid.record_separate_passes(&mut encoder);
                gpu.queue.submit([encoder.finish()]);
                let separate = gpu.read_chain(pyramid.texture(), pyramid.sizes());
                assert_eq!(one, separate, "{reduction:?} {strategy:?} depth={depth}");
            }
        }
    }
}
