//! Where a multi-token pass runs when decode is on a device —
//! `[orangu-server].prefill_backend`: `device` (with decode), `cpu` (the
//! CPU backend, while single-token decode stays on the device), or
//! `auto` (whichever does a prompt-shaped GEMM faster, measured once at
//! load). `ORANGU_HYBRID_PREFILL_CPU=1` is `cpu` from the environment.
//! Set by `main` before the model is built; read by the architectures
//! that can split a pass that way when they load
//! (`arch::qwen_hybrid::Trunk`).

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Instant;

use super::backend::{Backend, CpuBackend};
use super::loader::QuantMatrix;

/// The configured choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PrefillBackend {
    /// Prompts run where decode runs.
    #[default]
    Device,
    /// Prompts run on the CPU backend.
    Cpu,
    /// Measured at load: one prompt-shaped GEMM on each backend, the
    /// faster one takes the prompts.
    Auto,
}

static CHOICE: AtomicU8 = AtomicU8::new(0);

pub fn set(choice: PrefillBackend) {
    CHOICE.store(choice as u8, Ordering::Relaxed);
}

pub fn choice() -> PrefillBackend {
    if super::env::flag_on("ORANGU_HYBRID_PREFILL_CPU") {
        return PrefillBackend::Cpu;
    }
    match CHOICE.load(Ordering::Relaxed) {
        1 => PrefillBackend::Cpu,
        2 => PrefillBackend::Auto,
        _ => PrefillBackend::Device,
    }
}

/// The tokens the probe's GEMM carries: a short prompt, past the point
/// where a device switches to its batched kernels.
const PROBE_TOKENS: usize = 64;

/// The CPU backend when prompts should run there rather than on
/// `device`, by the configured choice — for `auto`, by timing `w` (a
/// projection of the model, the FFN gate's shape is the right one) at
/// [`PROBE_TOKENS`] tokens on each. `None` when `device` *is* the CPU
/// backend, or when prompts stay with it.
pub fn cpu_for_prompts(device: &Arc<dyn Backend>, w: &QuantMatrix) -> Option<Arc<dyn Backend>> {
    device.as_wgpu()?;
    let cpu: Arc<dyn Backend> = Arc::new(CpuBackend);
    match choice() {
        PrefillBackend::Device => None,
        PrefillBackend::Cpu => Some(cpu),
        PrefillBackend::Auto => {
            let on_device = time_gemm(device.as_ref(), w);
            let on_cpu = time_gemm(cpu.as_ref(), w);
            let winner = if on_cpu < on_device { "cpu" } else { "device" };
            eprintln!(
                "orangu-server: [prefill] auto: a {PROBE_TOKENS}-token {}x{} GEMM takes {:.1} ms on the device, {:.1} ms on the cpu — prompts on the {winner}",
                w.in_dim,
                w.out_dim,
                on_device * 1e3,
                on_cpu * 1e3,
            );
            (on_cpu < on_device).then_some(cpu)
        }
    }
}

/// Seconds for one `matmul` of `w` at [`PROBE_TOKENS`] tokens on
/// `backend`: the best of three after a warm-up (a device's first call
/// builds its pipelines and lifts its clock).
fn time_gemm(backend: &dyn Backend, w: &QuantMatrix) -> f64 {
    let x: Vec<f32> = (0..w.in_dim * PROBE_TOKENS)
        .map(|i| ((i * 7919) % 257) as f32 / 257.0 - 0.5)
        .collect();
    let _ = backend.matmul(&x, PROBE_TOKENS, w);
    (0..3)
        .map(|_| {
            let t = Instant::now();
            let _ = backend.matmul(&x, PROBE_TOKENS, w);
            t.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}
