use wgpu_depth_pyramid::{mip_sizes, reference, Reduction, Viewport};

#[test]
fn folded_edges_are_a_partition() {
    for h in 1..=65 {
        for w in 1..=65 {
            let values: Vec<_> = (0..w * h)
                .map(|i| (i + 1) as f32 / (w * h) as f32)
                .collect();
            let viewport = Viewport {
                offset: [0, 0],
                size: [w, h],
            };
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
        Viewport {
            offset: [0, 0],
            size: [5, 3],
        },
        Reduction::Min,
    )
    .unwrap();
    assert_eq!(chain[1].texels, [0.4, 0.0]);
    let cropped = reference::build(
        &values,
        [5, 3],
        Viewport {
            offset: [1, 1],
            size: [3, 2],
        },
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
    let vp = Viewport {
        offset: [0, 0],
        size: [3, 1],
    };
    assert_eq!(
        reference::build(&values, [3, 1], vp, Reduction::Min).unwrap()[1].texels[0].to_bits(),
        1
    );
    assert!(mip_sizes([0, 1]).is_err());
    assert!(reference::build(&[f32::NAN; 3], [3, 1], vp, Reduction::Min).is_err());
    assert!(reference::build(
        &values,
        [3, 1],
        Viewport {
            offset: [u32::MAX, 0],
            size: [2, 1]
        },
        Reduction::Min
    )
    .is_err());
    assert!(reference::build(&values, [4, 1], vp, Reduction::Min).is_err());
}
