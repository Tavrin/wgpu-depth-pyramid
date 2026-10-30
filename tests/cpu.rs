#[path = "../examples/support/ao_bounds.rs"]
mod ao_bounds;
use wgpu_depth_pyramid::{mip_sizes, reference, Reduction, Viewport};

#[test]
fn folded_edges_are_a_partition() {
    for h in 1..=65 {
        for w in 1..=65 {
            let values: Vec<_> = (0..w * h)
                .map(|i| (i + 1) as f32 / (w * h) as f32)
                .collect();
            let viewport = Viewport::new([0, 0], [w, h]);
            for reduction in [Reduction::Min, Reduction::Max] {
                let chain = reference::build(&values, [w, h], viewport, reduction).unwrap();
                assert_eq!(
                    chain.iter().map(|m| m.size).collect::<Vec<_>>(),
                    mip_sizes([w, h]).unwrap()
                );
                let last = chain.last().unwrap().texels[0];
                assert_eq!(
                    last.to_bits(),
                    if reduction == Reduction::Min {
                        values[0]
                    } else {
                        *values.last().unwrap()
                    }
                    .to_bits()
                );
            }
        }
    }
}

#[test]
fn explicit_odd_coverage_and_viewport() {
    let values = [
        0.9, 0.8, 0.7, 0.6, 0.1, 0.5, 0.4, 0.3, 0.2, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let chain = reference::build(
        &values,
        [5, 3],
        Viewport::new([0, 0], [5, 3]),
        Reduction::Min,
    )
    .unwrap();
    assert_eq!(chain[1].texels, [0.4, 0.0]);
    let cropped = reference::build(
        &values,
        [5, 3],
        Viewport::new([1, 1], [3, 2]),
        Reduction::Max,
    )
    .unwrap();
    assert_eq!(cropped[0].texels, [0.4, 0.3, 0.2, 1.0, 1.0, 1.0]);
    assert_eq!(cropped.last().unwrap().texels, [1.0]);
}

#[test]
fn subnormals_and_invalid_input() {
    let tiny = f32::from_bits(1);
    let values = [tiny, f32::from_bits(0x007fffff), tiny];
    let vp = Viewport::new([0, 0], [3, 1]);
    assert_eq!(
        reference::build(&values, [3, 1], vp, Reduction::Min).unwrap()[1].texels[0].to_bits(),
        1
    );
    assert!(mip_sizes([0, 1]).is_err());
    assert!(reference::build(&[f32::NAN; 3], [3, 1], vp, Reduction::Min).is_err());
    assert!(reference::build(
        &values,
        [3, 1],
        Viewport::new([u32::MAX, 0], [2, 1]),
        Reduction::Min
    )
    .is_err());
    assert!(reference::build(&values, [4, 1], vp, Reduction::Min).is_err());
}

#[test]
fn ao_edges_parameters_and_known_results() {
    let average = Reduction::Ao {
        effect_radius: 4.0,
        effect_falloff_range: 0.0,
    };
    let data = [
        0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    let chain = reference::build(&data, [5, 3], Viewport::new([0, 0], [5, 3]), average).unwrap();
    // Last row and column are intentionally dropped, unlike min/max folding.
    assert_eq!(
        chain[1]
            .texels
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        [0.125f32.to_bits(), 0.625f32.to_bits()]
    );
    assert_eq!(chain[2].texels[0].to_bits(), 0.375f32.to_bits());
    let thin = reference::build(
        &[0.25, 0.75, 1.0],
        [1, 3],
        Viewport::new([0, 0], [1, 3]),
        average,
    )
    .unwrap();
    assert_eq!(thin[1].texels[0].to_bits(), 0.5f32.to_bits());
    for (radius, falloff) in [
        (f32::NAN, 0.5),
        (f32::INFINITY, 0.5),
        (-1.0, 0.5),
        (1.0, f32::NAN),
        (1.0, -0.1),
        (1.0, 1.1),
        (f32::MAX, 0.0),
    ] {
        assert!(reference::build(
            &[0.5; 4],
            [2, 2],
            Viewport::new([0, 0], [2, 2]),
            Reduction::Ao {
                effect_radius: radius,
                effect_falloff_range: falloff
            }
        )
        .is_err());
    }
    for radius in [0.0, 0.001, 0.5, 2.0] {
        let r = Reduction::Ao {
            effect_radius: radius,
            effect_falloff_range: 0.615,
        };
        let chain =
            reference::build(&[0.25; 15], [5, 3], Viewport::new([0, 0], [5, 3]), r).unwrap();
        assert!(chain
            .iter()
            .flat_map(|m| &m.texels)
            .all(|v| v.to_bits() == 0.25f32.to_bits()));
    }
}

#[test]
fn ao_arithmetic_flushes_subnormals_but_copy_preserves_bits() {
    for bits in [1, 0x007fffff] {
        let chain = reference::build(
            &[f32::from_bits(bits); 2],
            [1, 2],
            Viewport::new([0, 0], [1, 2]),
            Reduction::Ao {
                effect_radius: 0.5,
                effect_falloff_range: 0.615,
            },
        )
        .unwrap();
        assert!(chain[0].texels.iter().all(|v| v.to_bits() == bits));
        assert_eq!(chain[1].texels[0].to_bits(), 0);
    }
}

#[test]
fn ao_forward_bounds_cover_division_rounding_and_detect_wrong_filter() {
    let reduction = Reduction::Ao {
        effect_radius: 0.5,
        effect_falloff_range: 0.615,
    };
    // Independent reciprocal-rounding counterexample: CPU result ...2441,
    // native reference shader ...2443. These are fixtures, never lookup data.
    let data = [1060205814, 1062833601, 1060737293, 1058962017].map(f32::from_bits);
    let cpu = reference::build(&data, [2, 2], Viewport::new([0, 0], [2, 2]), reduction).unwrap();
    assert_eq!(cpu[1].texels[0].to_bits(), 1060872441);
    let bounds = ao_bounds::build(&cpu, reduction);
    let tolerance = bounds[1][0].tolerance(cpu[1].texels[0]);
    let delta = (f64::from(f32::from_bits(1060872443)) - f64::from(cpu[1].texels[0])).abs();
    assert!(delta <= tolerance);
    assert!(tolerance < 0.00001);
    let average = data.iter().map(|&v| f64::from(v)).sum::<f64>() / 4.0;
    assert!((average - f64::from(cpu[1].texels[0])).abs() > tolerance);
    for size in [[1, 65], [65, 1], [7, 9], [65, 33]] {
        let data: Vec<_> = (0..size[0] * size[1])
            .map(|i| (i % 127) as f32 / 127.0)
            .collect();
        for (radius, falloff) in [
            (0.0, 0.0),
            (0.001, 1.0),
            (0.5, 0.615),
            (2.0, 0.1),
            (0.5, 0.0),
        ] {
            let r = Reduction::Ao {
                effect_radius: radius,
                effect_falloff_range: falloff,
            };
            let cpu = reference::build(&data, size, Viewport::new([0, 0], size), r).unwrap();
            let bounds = ao_bounds::build(&cpu, r);
            for (mip, b) in cpu.iter().zip(&bounds) {
                for (&value, &b) in mip.texels.iter().zip(b) {
                    assert!(b.tolerance(value).is_finite());
                }
            }
        }
    }
}

#[test]
fn ao_bound_retains_information_through_constant_deep_chains() {
    for (radius, falloff) in [(0.0, 0.0), (0.5, 0.615), (0.001, 1.0)] {
        let reduction = Reduction::Ao {
            effect_radius: radius,
            effect_falloff_range: falloff,
        };
        let cpu = reference::build(
            &[1.0; 4097],
            [1, 4097],
            Viewport::new([0, 0], [1, 4097]),
            reduction,
        )
        .unwrap();
        let bounds = ao_bounds::build(&cpu, reduction);
        let last = cpu.len() - 1;
        let tolerance = bounds[last][0].tolerance(cpu[last].texels[0]);
        assert!(tolerance < 0.0001, "deep constant chain bound {tolerance}");
        assert!((1.001 - f64::from(cpu[last].texels[0])).abs() > tolerance);
    }
}
