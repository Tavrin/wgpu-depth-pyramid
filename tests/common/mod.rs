pub fn sizes() -> Vec<[u32; 2]> {
    let mut cases = vec![
        [1, 1],
        [1, 2],
        [2, 1],
        [1, 4097],
        [4097, 1],
        [2, 3],
        [3, 2],
        [3, 5],
        [7, 9],
        [8, 8],
        [9, 15],
        [15, 17],
        [16, 16],
        [17, 31],
        [31, 33],
        [32, 32],
        [33, 63],
        [63, 65],
        [64, 64],
        [65, 127],
        [127, 129],
        [128, 128],
        [129, 255],
        [255, 257],
        [257, 129],
        [1920, 1080],
        [2560, 1440],
        [3840, 2160],
        [4097, 2049],
    ];
    // Random small rectangles exercise unequal folding remainders at every fused boundary.
    let mut state = 12345u32;
    for _ in 0..32 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        cases.push([1 + state % 300, 1 + (state >> 16) % 300]);
    }
    cases
}
