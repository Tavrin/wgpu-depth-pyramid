#![forbid(unsafe_code)]
mod support;
#[path = "support/workload.rs"]
mod workload;
fn main() {
    workload::run();
}
