#![cfg(not(target_arch = "wasm32"))]
#[path = "../examples/support/ao_bounds.rs"]
mod ao_bounds;
mod common;
#[path = "../examples/support/naive.rs"]
mod naive;
#[path = "../examples/support/mod.rs"]
mod support;
use wgpu_depth_pyramid::{reference, PyramidKernel, Reduction, Strategy, Viewport};

const SETTINGS: [Reduction; 5] = [
    Reduction::Ao {
        effect_radius: 0.5,
        effect_falloff_range: 0.615,
    },
    Reduction::Ao {
        effect_radius: 0.0,
        effect_falloff_range: 0.0,
    },
    Reduction::Ao {
        effect_radius: 0.001,
        effect_falloff_range: 1.0,
    },
    Reduction::Ao {
        effect_radius: 2.0,
        effect_falloff_range: 0.1,
    },
    Reduction::Ao {
        effect_radius: 0.5,
        effect_falloff_range: 0.0,
    },
];

#[test]
#[ignore = "requires a hardware GPU under an exclusive GPU lease"]
fn ao_cpu_tolerance_and_reference_shader_bit_exact() {
    run_matrix();
}

fn run_matrix() {
    let oracle = "CPU-WGSL-bound+GPU-bitwise";
    let gpu = support::Gpu::new(false);
    let kernel = PyramidKernel::new(&gpu.device);
    let mut chains = 0u64;
    let mut comparisons = 0u64;
    let mut max_error = [0.0f64; 32];
    let mut max_bound = [0.0f64; 32];
    let mut max_fraction = [0.0f64; 32];
    let mut levels = 0;
    for size in common::sizes() {
        for offset in [[0, 0], [3, 5]] {
            let source_size = [size[0] + offset[0], size[1] + offset[1]];
            let viewport = Viewport::new(offset, size);
            for pattern in 0..10 {
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
                    // Full existing size/pattern/format/offset matrix for default parameters;
                    // additional parameter extremes on small and degenerate inputs.
                    let settings = if size[0] * size[1] <= 4097 {
                        &SETTINGS[..]
                    } else {
                        &SETTINGS[..1]
                    };
                    for &reduction in settings {
                        let expected =
                            reference::build(&input, source_size, viewport, reduction).unwrap();
                        let bounds = ao_bounds::build(&expected, reduction);
                        let tolerances: Vec<Vec<_>> = bounds
                            .iter()
                            .zip(&expected)
                            .map(|(b, m)| {
                                b.iter()
                                    .zip(&m.texels)
                                    .map(|(&b, &v)| b.tolerance(v))
                                    .collect()
                            })
                            .collect();
                        drop(bounds);
                        let baseline =
                            naive::Naive::with_viewport(&gpu.device, &source, viewport, reduction);
                        let mut strategies = vec![Strategy::Auto];
                        if size[0] * size[1] <= 70000 {
                            strategies.extend([
                                Strategy::Naive,
                                Strategy::Merged,
                                Strategy::Legacy,
                            ]);
                            for tile in [32, 64] {
                                for depth in 2..=if tile == 32 { 5 } else { 6 } {
                                    strategies.push(Strategy::Fused { tile, depth });
                                    strategies.push(Strategy::FusedMerged { tile, depth });
                                }
                            }
                        }
                        let mut encoder = gpu.device.create_command_encoder(&Default::default());
                        baseline.record(&mut encoder);
                        gpu.queue.submit([encoder.finish()]);
                        let reference_gpu = gpu.read_chain(&baseline.texture, &baseline.sizes);
                        levels = levels.max(reference_gpu.len());
                        for (level, actual) in reference_gpu.iter().enumerate().skip(1) {
                            for (i, &a) in actual.iter().enumerate() {
                                let error = (f64::from(f32::from_bits(a))
                                    - f64::from(expected[level].texels[i]))
                                .abs();
                                let bound = tolerances[level][i];
                                max_error[level] = max_error[level].max(error);
                                max_bound[level] = max_bound[level].max(bound);
                                if bound > 0.0 {
                                    max_fraction[level] = max_fraction[level].max(error / bound);
                                }
                            }
                        }
                        for strategy in strategies {
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
                            let actual = gpu.read_chain(pyramid.texture(), pyramid.sizes());
                            assert_eq!(pyramid.sizes(), baseline.sizes.as_slice());
                            assert_eq!(actual.len(), reference_gpu.len());
                            assert_eq!(actual.len(), expected.len());
                            for (level, (actual, reference)) in
                                actual.iter().zip(&reference_gpu).enumerate()
                            {
                                assert_eq!(actual.len(), reference.len());
                                assert_eq!(actual.len(), expected[level].texels.len());
                                for (i, (&a, &r)) in actual.iter().zip(reference).enumerate() {
                                    assert_eq!(a,r,"GPU reference {size:?} {offset:?} pattern={pattern} depth={depth} {reduction:?} {strategy:?} mip={level} texel={i}");
                                    let cpu = expected[level].texels[i];
                                    if level == 0 {
                                        assert_eq!(a, cpu.to_bits(), "mip-zero CPU identity");
                                    } else {
                                        let error =
                                            (f64::from(f32::from_bits(a)) - f64::from(cpu)).abs();
                                        let tolerance = tolerances[level][i];
                                        assert!(error <= tolerance, "CPU reference {size:?} {offset:?} pattern={pattern} depth={depth} {reduction:?} {strategy:?} mip={level} texel={i}: error={error} bound={tolerance}");
                                    }
                                }
                                comparisons += actual.len() as u64;
                            }
                            chains += 1;
                        }
                    }
                }
            }
        }
        eprintln!("AO PASS oracle={oracle} size={size:?} chains={chains} texels={comparisons}");
    }
    eprintln!("AO TOTAL oracle={oracle} chains={chains} texels={comparisons}");
    for level in 1..levels {
        eprintln!(
            "AO ERROR mip={level} max_absolute={} max_bound={} max_error_over_own_bound={}",
            max_error[level], max_bound[level], max_fraction[level]
        );
    }
}
