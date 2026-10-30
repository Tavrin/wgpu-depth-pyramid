#![forbid(unsafe_code)]
#[path = "support/ao_bounds.rs"]
mod ao_bounds;
#[path = "support/naive.rs"]
mod naive;
mod support;
use std::time::Instant;
use wgpu_depth_pyramid::{reference, Pyramid, PyramidKernel, Reduction, Strategy, Viewport};
const FRAMES: u32 = 40;
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
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
}
fn queries(gpu: &support::Gpu, count: u32) -> wgpu::QuerySet {
    gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: None,
        ty: wgpu::QueryType::Timestamp,
        count,
    })
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
        .map(|q| u64::from_le_bytes(*q))
        .collect()
}
fn profile(gpu: &support::Gpu, pyramid: &Pyramid, size: [u32; 2], reduction: Reduction) {
    let n = pyramid.dispatch_count() as u32;
    let queries = queries(gpu, FRAMES * 2 * n);
    for frame in 0..FRAMES {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        pyramid.record_profiled(&mut encoder, &queries, frame * 2 * n);
        gpu.queue.submit([encoder.finish()]);
    }
    let ticks = read_queries(gpu, &queries, FRAMES * 2 * n);
    let scale = f64::from(gpu.queue.get_timestamp_period()) / 1000.;
    for step in 0..n {
        let mut times = Vec::new();
        for frame in 0..FRAMES {
            let i = ((frame * n + step) * 2) as usize;
            let us = (ticks[i + 1] - ticks[i]) as f64 * scale;
            times.push(us);
            eprintln!(
                "dispatch_sample,{size:?},{reduction:?},{:?},{step},{frame},{},{},{us:.6}",
                pyramid.strategy(),
                ticks[i],
                ticks[i + 1]
            );
        }
        println!(
            "dispatch_separate_passes,{size:?},{reduction:?},{:?},{step},{:.3}",
            pyramid.strategy(),
            median(&mut times)
        );
    }
}
fn main() {
    let gpu = support::Gpu::new(true);
    let kernel = PyramidKernel::new(&gpu.device);
    let ao_run = std::env::args().any(|arg| arg == "--ao");
    let verification = if ao_run {
        "CPU-WGSL-bound+GPU-bitwise"
    } else {
        "CPU-bitwise+GPU-bitwise"
    };
    println!("verification={verification}");
    eprintln!("verification={verification}");
    let final_run = std::env::args().any(|arg| arg == "--final");
    println!(
        "adapter={:?}; default_limits={:?}",
        gpu.info,
        wgpu::Limits::default()
    );
    println!("timestamp_period_ns={}; frames=40; warmups=8; rotating/reversing three arms; full chain including mip0",gpu.queue.get_timestamp_period());
    println!("row,size,reduction,strategy,dispatches,default_gpu_us,naive_gpu_us,scalar_gpu_us,default_over_naive,default_over_scalar,default_cpu_us,naive_cpu_us,scalar_cpu_us");
    let mut strategies = vec![Strategy::Legacy, Strategy::Naive, Strategy::Merged];
    for tile in [32, 64] {
        for depth in 2..=if tile == 32 { 5 } else { 6 } {
            strategies.push(Strategy::Fused { tile, depth });
            strategies.push(Strategy::FusedMerged { tile, depth });
        }
    }
    if final_run {
        strategies = vec![Strategy::Auto];
    }
    for size in [[1920, 1080], [2560, 1440], [3840, 2160]] {
        let data = support::depths(size, 0);
        let source = gpu.source(size, &data, false);
        let reductions = if ao_run {
            vec![Reduction::Ao {
                effect_radius: 0.5,
                effect_falloff_range: 0.615,
            }]
        } else {
            vec![Reduction::Min, Reduction::Max]
        };
        for reduction in reductions {
            let viewport = Viewport::new([0, 0], size);
            let cpu_expected = reference::build(&data, size, viewport, reduction).unwrap();
            let bounds = ao_run.then(|| ao_bounds::build(&cpu_expected, reduction));
            let library_naive = kernel
                .prepare_with_strategy(&gpu.device, &source, viewport, reduction, Strategy::Naive)
                .unwrap();
            let baseline = naive::Naive::new(&gpu.device, &source, reduction);
            for &strategy in &strategies {
                let pyramid = kernel
                    .prepare_with_strategy(&gpu.device, &source, viewport, reduction, strategy)
                    .unwrap();
                let record = |arm: u32, encoder: &mut wgpu::CommandEncoder| match arm {
                    0 => pyramid.record(encoder),
                    1 => baseline.record(encoder),
                    _ => library_naive.record(encoder),
                };
                let mut encoder = gpu.device.create_command_encoder(&Default::default());
                for arm in 0..3 {
                    record(arm, &mut encoder);
                }
                baseline.record(&mut encoder);
                library_naive.record(&mut encoder);
                gpu.queue.submit([encoder.finish()]);
                let expected = gpu.read_chain(&baseline.texture, &baseline.sizes);
                for actual in [
                    gpu.read_chain(pyramid.texture(), pyramid.sizes()),
                    gpu.read_chain(&baseline.texture, &baseline.sizes),
                    gpu.read_chain(library_naive.texture(), library_naive.sizes()),
                ] {
                    assert_eq!(actual.len(), expected.len());
                    for (level, (a, e)) in actual.iter().zip(&expected).enumerate() {
                        assert_eq!(a.len(), e.len());
                        for (i, (&a, &e)) in a.iter().zip(e).enumerate() {
                            assert_eq!(
                                a, e,
                                "{size:?} {reduction:?} {strategy:?} mip={level} texel={i}"
                            );
                            let cpu = cpu_expected[level].texels[i];
                            if let Some(bounds) = &bounds {
                                let error = (f64::from(f32::from_bits(a)) - f64::from(cpu)).abs();
                                assert!(
                                    error <= bounds[level][i].tolerance(cpu),
                                    "AO CPU tolerance mip={level} texel={i}"
                                );
                            } else {
                                assert_eq!(a, cpu.to_bits(), "min/max CPU identity");
                            }
                        }
                    }
                }
                for frame in 0..8 {
                    let mut encoder = gpu.device.create_command_encoder(&Default::default());
                    for k in 0..3 {
                        record((frame + k) % 3, &mut encoder);
                    }
                    gpu.queue.submit([encoder.finish()]);
                }
                gpu.device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .unwrap();
                let queries = queries(&gpu, FRAMES * 6);
                let mut cpu: [Vec<f64>; 3] = Default::default();
                for frame in 0..FRAMES {
                    let mut encoder = gpu.device.create_command_encoder(&Default::default());
                    let order = if frame % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] };
                    for k in 0..3 {
                        let arm = order[((k + frame / 2) % 3) as usize];
                        let index = frame * 6 + arm * 2;
                        marker(&mut encoder, &queries, index, true);
                        let start = Instant::now();
                        record(arm, &mut encoder);
                        let elapsed = start.elapsed().as_secs_f64() * 1e6;
                        cpu[arm as usize].push(elapsed);
                        marker(&mut encoder, &queries, index + 1, false);
                    }
                    gpu.queue.submit([encoder.finish()]);
                }
                let ticks = read_queries(&gpu, &queries, FRAMES * 6);
                let scale = f64::from(gpu.queue.get_timestamp_period()) / 1000.;
                let mut times: [Vec<f64>; 3] = Default::default();
                for frame in 0..FRAMES {
                    for arm in 0..3 {
                        let i = (frame * 6 + arm * 2) as usize;
                        assert!(ticks[i + 1] > ticks[i]);
                        let us = (ticks[i + 1] - ticks[i]) as f64 * scale;
                        times[arm as usize].push(us);
                        eprintln!(
                            "sample,{size:?},{reduction:?},{:?},{frame},{arm},{},{},{us:.6},{:.6}",
                            pyramid.strategy(),
                            ticks[i],
                            ticks[i + 1],
                            cpu[arm as usize][frame as usize]
                        );
                    }
                }
                let med = times.each_mut().map(|v| median(v));
                let host = cpu.each_mut().map(|v| median(v));
                println!("row,{size:?},{reduction:?},{:?},{},{:.3},{:.3},{:.3},{:.4},{:.4},{:.3},{:.3},{:.3}",pyramid.strategy(),pyramid.dispatch_count(),med[0],med[1],med[2],med[0]/med[1],med[0]/med[2],host[0],host[1],host[2]);
                if matches!(
                    strategy,
                    Strategy::Legacy
                        | Strategy::Auto
                        | Strategy::Fused { tile: 64, depth: 4 }
                        | Strategy::Naive
                ) {
                    profile(&gpu, &pyramid, size, reduction);
                }
            }
        }
    }
}
