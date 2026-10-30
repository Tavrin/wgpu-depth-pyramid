//! Independent CPU oracle for nonnegative finite depth values in [0, 1].
use crate::{mip_sizes, validate, Error, Reduction, Viewport};

/// One tightly packed CPU mip level.
#[derive(Debug)]
pub struct Mip {
    /// Width and height in texels.
    pub size: [u32; 2],
    /// Row-major depth values without padding.
    pub texels: Vec<f32>,
}

/// Build a tightly packed chain. Select on raw unsigned encodings: for nonnegative
/// finite floats their ordering is the numerical ordering and subnormals survive.
/// Negative zero, negative values, infinities and NaNs are outside the input contract.
pub fn build(
    source: &[f32],
    source_size: [u32; 2],
    viewport: Viewport,
    reduction: Reduction,
) -> Result<Vec<Mip>, Error> {
    validate(source_size, viewport)?;
    if u64::from(source_size[0]) * u64::from(source_size[1]) != source.len() as u64 {
        return Err(Error("source length does not match dimensions"));
    }
    if source.iter().any(|v| v.to_bits() > 1.0f32.to_bits()) {
        return Err(Error(
            "depth must be finite, nonnegative and at most one; negative zero is excluded",
        ));
    }
    let sizes = mip_sizes(viewport.size)?;
    let mut base = Vec::with_capacity(viewport.size[0] as usize * viewport.size[1] as usize);
    for y in viewport.offset[1]..viewport.offset[1] + viewport.size[1] {
        let start = y as usize * source_size[0] as usize + viewport.offset[0] as usize;
        base.extend_from_slice(&source[start..start + viewport.size[0] as usize]);
    }
    let mut mips = vec![Mip {
        size: viewport.size,
        texels: base,
    }];
    for &size in &sizes[1..] {
        let parent = mips.last().unwrap();
        let mut texels = Vec::with_capacity(size[0] as usize * size[1] as usize);
        for y in 0..size[1] {
            let end_y = if y + 1 == size[1] {
                parent.size[1]
            } else {
                2 * y + 2
            };
            for x in 0..size[0] {
                let end_x = if x + 1 == size[0] {
                    parent.size[0]
                } else {
                    2 * x + 2
                };
                let mut value = parent.texels[(2 * y * parent.size[0] + 2 * x) as usize].to_bits();
                for sy in 2 * y..end_y {
                    for sx in 2 * x..end_x {
                        let sample = parent.texels[(sy * parent.size[0] + sx) as usize].to_bits();
                        value = match reduction {
                            Reduction::Min => value.min(sample),
                            Reduction::Max => value.max(sample),
                        };
                    }
                }
                texels.push(f32::from_bits(value));
            }
        }
        mips.push(Mip { size, texels });
    }
    Ok(mips)
}
