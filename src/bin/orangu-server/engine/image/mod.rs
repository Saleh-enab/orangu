// Copyright (C) 2026 The orangu community
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Text-to-image and image-to-image with Qwen-Image — the pipeline a
//! `qwen_image` GGUF is served through.
//!
//! A `qwen_image` file is one third of a model. It holds the diffusion
//! transformer ([`transformer`]), which denoises a *latent* picture under
//! the guidance of a prompt's hidden states. Those hidden states come from a
//! Qwen2.5-VL-7B text encoder, an ordinary `qwen2vl` GGUF served through
//! the usual `arch::llama` path and read through
//! `ModelForward::forward_hidden_states` — the same call an embeddings
//! request makes, which is why the whole language-model engine is reused
//! rather than a second encoder written. And the latent becomes pixels
//! through the Qwen-Image VAE ([`vae`]), a `safetensors` file. The server
//! finds the two companions in the models directory ([`Companions`]), or
//! where the configuration points.
//!
//! Generation is diffusers' `QwenImagePipeline`, step for step:
//! encode the prompt (and the negative one), start from seeded Gaussian
//! noise (or from the attached picture's latent, noised to the chosen
//! `strength`), and walk the flow-matching schedule ([`scheduler`]) — at
//! every step one transformer pass per prompt, classifier-free guidance
//! combining the two with the norm-preserving rescale Qwen-Image uses, one
//! Euler step — then decode and write PNG or JPEG ([`codec`]).
//!
//! One picture at a time: the transformer is 20 billion parameters and a
//! step is a full pass over them, so two requests interleaved would be
//! slower than two in turn. Concurrent callers queue on the pipeline's lock.

pub mod codec;
pub mod lora;
pub mod safetensors;
pub mod scheduler;
pub mod transformer;
pub mod vae;

use crate::engine::arch::ModelForward;
use crate::engine::backend::Backend;
use crate::engine::loader::LoadedModel;
use crate::engine::tokenizer::Tokenizer;
use anyhow::{Context, Result, bail, ensure};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

pub use codec::ImageFormat;
use scheduler::{Noise, ScheduleConfig};
use transformer::{ForwardInput, QwenImageTransformer};
use vae::{Feature, LATENTS_MEAN, LATENTS_STD, QwenImageVae, SPATIAL_COMPRESSION, Z_DIM};

/// The text encoder's chat framing around a prompt — diffusers'
/// `prompt_template_encode`. The hidden states of the framing itself are
/// dropped ([`Pipeline::encode_prompt`]); only the prompt's own tokens
/// condition the picture, but they are encoded *in this context*.
const PROMPT_TEMPLATE_PREFIX: &str = "<|im_start|>system\nDescribe the image by detailing the \
     color, shape, size, texture, quantity, text, spatial relationships of the objects and \
     background:<|im_end|>\n<|im_start|>user\n";
const PROMPT_TEMPLATE_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";
/// diffusers' `tokenizer_max_length`: how much of a prompt is read.
const MAX_PROMPT_TOKENS: usize = 1024;

/// The architecture string of the text encoder the pipeline needs.
pub const TEXT_ENCODER_ARCHITECTURE: &str = "qwen2vl";
/// The width of its hidden states, which is what tells a Qwen2.5-VL-7B
/// apart from the 3B (2048) and 72B (8192) in a models directory.
pub const TEXT_ENCODER_WIDTH: u64 = 3584;

/// Where the pipeline gets its companions from — the same places
/// `orangu-server download` fetches them from beside a `qwen_image` model.
pub const TEXT_ENCODER_REPO: &str = orangu::model_download::QWEN_IMAGE_TEXT_ENCODER_REPO;
pub const TEXT_ENCODER_DEFAULT_TAG: &str = orangu::model_download::QWEN_IMAGE_TEXT_ENCODER_TAG;
pub const VAE_REPO: &str = orangu::model_download::QWEN_IMAGE_VAE_REPO;
pub const VAE_FILE: &str = orangu::model_download::QWEN_IMAGE_VAE_FILE;

/// The two files a `qwen_image` model cannot generate without.
#[derive(Debug, Clone)]
pub struct Companions {
    pub text_encoder: PathBuf,
    pub vae: PathBuf,
}

impl Companions {
    /// Finds both companions: the configured path or spec when there is
    /// one, otherwise the models directory is searched — for a `qwen2vl`
    /// GGUF of the right width (the largest, when there are several
    /// quantizations), and for the VAE by its tensors rather than its name.
    pub fn locate(
        models_dir: &Path,
        text_encoder: Option<&str>,
        vae: Option<&str>,
    ) -> Result<Self> {
        let text_encoder = match text_encoder {
            Some(spec) => {
                orangu::model_spec::resolve_load_target(models_dir, spec)
                    .with_context(|| format!("resolving text_encoder '{spec}'"))?
                    .0
            }
            None => find_text_encoder(models_dir)?,
        };
        let vae = match vae {
            Some(path) => {
                let path = PathBuf::from(path);
                let path = if path.is_absolute() {
                    path
                } else {
                    models_dir.join(path)
                };
                ensure!(path.is_file(), "vae {} is not a file", path.display());
                path
            }
            None => find_vae(models_dir)?,
        };
        Ok(Self { text_encoder, vae })
    }
}

fn find_text_encoder(models_dir: &Path) -> Result<PathBuf> {
    orangu::model_spec::find_qwen_image_text_encoder(models_dir).ok_or_else(|| {
        anyhow::anyhow!(
            "no Qwen2.5-VL-7B text encoder ({TEXT_ENCODER_ARCHITECTURE}, width \
             {TEXT_ENCODER_WIDTH}) found under {}. A qwen_image model is conditioned on that \
             encoder's hidden states; download one with `orangu-server download \
             {TEXT_ENCODER_REPO}:{TEXT_ENCODER_DEFAULT_TAG}` or point [orangu-server].text_encoder \
             at it",
            models_dir.display()
        )
    })
}

fn find_vae(models_dir: &Path) -> Result<PathBuf> {
    orangu::model_spec::find_qwen_image_vae(models_dir).ok_or_else(|| {
        anyhow::anyhow!(
            "no Qwen-Image VAE (a .safetensors with the Wan decoder) found under {}. It turns \
             the model's latents into pixels; `orangu-server download` of a qwen_image model \
             fetches {VAE_REPO}'s {VAE_FILE} beside it, or point [orangu-server].vae at the file",
            models_dir.display()
        )
    })
}

/// One linear of the transformer, timed on a device and on the CPU — the
/// measurement `backend = auto` makes before committing a picture pipeline
/// to a GPU.
///
/// A picture is prefill-shaped work, hundreds of tokens through every
/// linear, and whether a given GPU beats the CPU at it is a question about
/// the two kernels on the two chips, not about the GPU existing: on a
/// board whose integrated Mali shares the CPU's memory and whose CPU has
/// `i8mm`, the device ran a 256-token step 7.8× *slower* than the CPU.
/// So rather than a rule about device classes, the pipeline runs the same
/// linear both ways and keeps the faster. See [`calibrate`].
#[derive(Debug, Clone, Copy)]
pub struct Calibration {
    /// The tensor's shape, for the log line.
    pub in_dim: usize,
    pub out_dim: usize,
    pub n_tokens: usize,
    /// The second run on each: the first pays for pipeline compilation and
    /// the weight upload, which a picture pays once too but which is not
    /// the per-step cost being compared. `device` is `None` when no device
    /// was offered — an explicit `backend = cpu`, where only the estimate
    /// below is wanted.
    pub device: Option<std::time::Duration>,
    pub cpu: std::time::Duration,
}

impl Calibration {
    pub fn cpu_wins(&self) -> bool {
        self.device.is_none_or(|device| self.cpu <= device)
    }

    /// A first guess at the pipeline's rate in latent-token passes per
    /// second, from the winner's time on the calibration linear. A pass
    /// over one image token is about 6.8 G multiply-adds through the
    /// token-wide linears (`doc/PERF-IMAGE.md`, *Arithmetic*), and the
    /// calibration linear is `in_dim × out_dim` of them per token — so the
    /// ratio scales the measured time up to a whole pass. Attention adds
    /// to it at larger pictures, so this is a floor on the time, refined
    /// by the first real picture ([`Pipeline::rate`]).
    pub fn token_passes_per_second(&self) -> f64 {
        let winner = match self.device {
            Some(device) if device < self.cpu => device,
            _ => self.cpu,
        };
        let linear_macs = (self.in_dim * self.out_dim) as f64;
        let pass_seconds_per_token =
            winner.as_secs_f64() / self.n_tokens as f64 * (MACS_PER_TOKEN_PASS / linear_macs);
        1.0 / pass_seconds_per_token.max(1e-9)
    }
}

/// Multiply-adds per image token per transformer pass, the token-wide
/// linears of every block: `3 × 3072² + 3072² + 2 × 3072 × 12288` per block,
/// sixty blocks.
const MACS_PER_TOKEN_PASS: f64 = 6.8e9;

/// Which tensor [`calibrate`] times: the first block's image MLP input
/// projection, the widest linear a step runs and the largest share of it
/// (`mlp` is over half of every pass).
const CALIBRATION_TENSOR: &str = "transformer_blocks.0.img_mlp.net.0.proj.weight";
/// The token count [`calibrate`] runs — a 256×256 picture's worth, the
/// smallest size worth generating.
const CALIBRATION_TOKENS: usize = 256;

/// Times [`CALIBRATION_TENSOR`] through `device` (when one is offered) and
/// through the CPU at [`CALIBRATION_TOKENS`] tokens, on synthetic
/// activations. Each is run twice and the second timed; the whole thing is
/// well under a second on the CPU and a few seconds on a slow device — a
/// startup cost that saves hours when it says no, and that seeds the wait
/// estimate a picture is announced with.
pub fn calibrate(transformer: &LoadedModel, device: Option<&dyn Backend>) -> Result<Calibration> {
    let w = transformer
        .matrix(CALIBRATION_TENSOR)
        .with_context(|| format!("qwen_image calibration tensor {CALIBRATION_TENSOR}"))?;
    let n_tokens = CALIBRATION_TOKENS;
    let x: Vec<f32> = (0..n_tokens * w.in_dim)
        .map(|i| ((i * 37 % 23) as f32 - 11.0) * 0.031)
        .collect();
    let time = |backend: &dyn Backend| -> std::time::Duration {
        let op = crate::engine::backend::MatmulOp {
            x: &x,
            n_tokens,
            w: &w,
        };
        let _ = backend.matmul_batch(std::slice::from_ref(&op));
        let started = Instant::now();
        let _ = backend.matmul_batch(std::slice::from_ref(&op));
        started.elapsed()
    };
    Ok(Calibration {
        in_dim: w.in_dim,
        out_dim: w.out_dim,
        n_tokens,
        device: device.map(time),
        cpu: time(&crate::engine::backend::CpuBackend),
    })
}

/// Where `[orangu-server].image_lora` points: a path as given, a path
/// under the models directory, or a `<user>/<repo>:<file>` reference
/// resolved through the hub-cache layout `download` writes
/// (`models--<user>--<repo>/snapshots/<commit>/<file>`).
pub fn resolve_lora(models_dir: &Path, spec: &str) -> Result<PathBuf> {
    let direct = PathBuf::from(spec);
    if direct.is_file() {
        return Ok(direct);
    }
    let under_models = models_dir.join(spec);
    if under_models.is_file() {
        return Ok(under_models);
    }
    if let Some((repo, file)) = spec.split_once(':')
        && repo.matches('/').count() == 1
    {
        let snapshots = models_dir
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("snapshots");
        if let Ok(entries) = std::fs::read_dir(&snapshots) {
            for entry in entries.flatten() {
                let candidate = entry.path().join(file);
                if candidate.is_file() {
                    return Ok(candidate);
                }
            }
        }
        bail!(
            "image_lora {spec}: {file} is not under {}; fetch it with `orangu-server download {spec}`",
            snapshots.display()
        );
    }
    bail!(
        "image_lora {spec}: no such file, as given or under {}",
        models_dir.display()
    )
}

/// Server-wide defaults a request may override — `[orangu-server].image_*`.
#[derive(Debug, Clone)]
pub struct ImageDefaults {
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub cfg_scale: f32,
    pub negative_prompt: String,
    pub strength: f32,
    /// The container a picture comes back in when the request names none
    /// and no attachment sets it.
    pub format: ImageFormat,
}

impl Default for ImageDefaults {
    /// Qwen-Image's own release settings: 1024x1024, 50 steps, true CFG 4.0
    /// against a blank negative prompt, and diffusers' image-to-image
    /// `strength` of 0.6.
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            steps: 50,
            cfg_scale: 4.0,
            negative_prompt: " ".to_string(),
            strength: 0.6,
            format: ImageFormat::Png,
        }
    }
}

/// One generation.
#[derive(Debug, Clone)]
pub struct ImageRequest {
    pub prompt: String,
    /// `None` takes the server default; `Some("")` is a blank prompt too.
    pub negative_prompt: Option<String>,
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    /// `<= 1` runs the positive prompt alone — half the work, no guidance.
    pub cfg_scale: f32,
    /// `None` draws one, and the result reports which.
    pub seed: Option<u64>,
    pub format: ImageFormat,
    /// A picture to start from, for image-to-image.
    pub init: Option<InitImage>,
}

#[derive(Debug, Clone)]
pub struct InitImage {
    /// PNG, JPEG or SVG bytes; resized to the request's size.
    pub bytes: Vec<u8>,
    /// How much of the schedule to run: `1.0` ignores the picture's content
    /// entirely (pure noise), `0.0` returns it unchanged.
    pub strength: f32,
}

#[derive(Debug, Clone)]
pub struct GeneratedImage {
    pub bytes: Vec<u8>,
    pub format: ImageFormat,
    pub width: usize,
    pub height: usize,
    pub seed: u64,
    pub steps: usize,
    pub elapsed: std::time::Duration,
    pub timings: Timings,
}

/// Where a generation's time went, phase by phase — what `orangu-bench
/// --image` reads to say which of the three models a slow picture is slow
/// in, and what the log line prints beside the total.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timings {
    /// Encoding the prompt (and the negative one, under guidance) through
    /// the text encoder — plus, for image-to-image, encoding the attached
    /// picture through the VAE.
    pub encode: std::time::Duration,
    /// The denoising loop: every transformer pass, both prompts when
    /// guided, and the Euler steps between them.
    pub denoise: std::time::Duration,
    /// Unpacking the latent, the VAE decode, and writing the file format.
    pub decode: std::time::Duration,
    /// The denoising loop's transformer passes by operation class — see
    /// [`transformer::Stages`].
    pub stages: transformer::Stages,
}

/// Where a generation is, for a progress bar. `step` is `0` for the
/// announcement before the first step, whose `seconds_per_step` is the
/// estimate from [`Pipeline::seconds_per_step`].
#[derive(Debug, Clone, Copy)]
pub struct Progress {
    /// Steps finished so far, and the total the request will run.
    pub step: usize,
    pub steps: usize,
    /// Seconds per step so far, for an estimate of what is left.
    pub seconds_per_step: f64,
}

/// What the pipeline has learned about its own speed, for saying how long
/// a picture will take before it starts.
///
/// A transformer pass is not one rate: the linears cost the same per token
/// at every size, attention costs more per token the more tokens there are
/// (it is quadratic), so a rate measured on a 256-token picture is
/// optimistic for a 4,096-token one by nearly 2×. The model here is the
/// two parts kept apart — seconds per token pass for everything but
/// attention, and seconds per token pass for attention *at the size it was
/// measured*, scaled by the ratio of sizes — plus the VAE decode per pixel
/// and the prompt encode, which the steps' rate never covered. Each
/// finished picture replaces all of it from its own `Timings`; the startup
/// calibration seeds the linear part and takes the rest from this board's
/// measured shares.
#[derive(Debug, Clone, Copy)]
pub struct RateModel {
    /// Seconds per latent-token pass, attention excluded.
    pub linear_per_token_pass: f64,
    /// Seconds per latent-token pass spent in attention, measured at
    /// `attention_tokens` tokens; at `n` tokens it is `× n /
    /// attention_tokens`.
    pub attention_per_token_pass: f64,
    pub attention_tokens: f64,
    /// Seconds of VAE decode per output pixel.
    pub decode_per_pixel: f64,
    /// Seconds of prompt encoding per picture.
    pub encode: f64,
}

/// The fewest latent tokens a picture may have to teach the [`RateModel`]:
/// below this the pass is weight-bound and says nothing about the per-token
/// rate a real picture runs at.
const RATE_MODEL_MIN_TOKENS: usize = 64;

impl RateModel {
    /// From the startup calibration alone: its rate is the linears'; the
    /// rest are this board's shares at 256 tokens — attention a tenth of
    /// the linears' time there, the VAE 30 µs a pixel, the encoder two
    /// seconds — refined by the first picture.
    pub fn from_calibration(token_passes_per_second: f64) -> Self {
        let linear = 1.0 / token_passes_per_second.max(1e-9);
        Self {
            linear_per_token_pass: linear,
            attention_per_token_pass: 0.1 * linear,
            attention_tokens: 256.0,
            decode_per_pixel: 30e-6,
            encode: 2.0,
        }
    }

    /// Seconds one denoising step takes at `n_tokens` latent tokens and
    /// `passes_per_step` transformer passes.
    pub fn seconds_per_step(&self, n_tokens: usize, passes_per_step: f64) -> f64 {
        let n = n_tokens as f64;
        let attention = self.attention_per_token_pass * n / self.attention_tokens.max(1.0);
        n * passes_per_step * (self.linear_per_token_pass + attention)
    }

    /// The whole picture: encode, the steps, the decode.
    pub fn seconds_for(&self, width: usize, height: usize, steps: usize, passes: f64) -> f64 {
        let n_tokens = (width / 16) * (height / 16);
        self.encode
            + self.seconds_per_step(n_tokens, passes) * steps as f64
            + self.decode_per_pixel * (width * height) as f64
    }

    /// Latent-token passes per second at `n_tokens`, for `/props`.
    pub fn token_passes_per_second(&self, n_tokens: usize) -> f64 {
        let per_step = self.seconds_per_step(n_tokens, 1.0);
        if per_step > 0.0 {
            n_tokens as f64 / per_step
        } else {
            0.0
        }
    }
}

impl Pipeline {
    /// The estimated seconds one denoising step takes for `n_tokens` latent
    /// tokens at `passes_per_step` transformer passes (two under guidance)
    /// — `None` before anything was measured.
    pub fn seconds_per_step(&self, n_tokens: usize, passes_per_step: f64) -> Option<f64> {
        self.rate_model()
            .map(|m| m.seconds_per_step(n_tokens, passes_per_step))
    }

    /// The current [`RateModel`], if any.
    pub fn rate_model(&self) -> Option<RateModel> {
        *self
            .rate
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The rate at the defaults' size, latent-token passes per second, for
    /// `/props`.
    pub fn token_passes_per_second(&self) -> Option<f64> {
        let d = self.defaults();
        self.rate_model()
            .map(|m| m.token_passes_per_second((d.width / 16) * (d.height / 16)))
    }

    /// The estimated wall time of a whole picture at the server defaults
    /// — the wait the console's user is looking at — or `None` before any
    /// rate is known.
    pub fn estimated_default_seconds(&self) -> Option<f64> {
        let d = self.defaults();
        let passes = if d.cfg_scale > 1.0 { 2.0 } else { 1.0 };
        self.rate_model()
            .map(|m| m.seconds_for(d.width, d.height, d.steps, passes))
    }
}

pub struct Pipeline {
    transformer: QwenImageTransformer,
    vae: QwenImageVae,
    /// The text encoder — the server's `ModelForward`, shared with the
    /// `Engine` that also answers embeddings with it.
    encoder: Arc<dyn ModelForward>,
    tokenizer: Arc<Tokenizer>,
    /// What a request gets when it does not say — the config's picture
    /// keys as the server came up, and since then whatever `POST /props`
    /// (the console's Image settings) last set. Read with [`Pipeline::
    /// defaults`], a clone: a request is built from one consistent set.
    defaults: RwLock<ImageDefaults>,
    /// The defaults as configured — what a reset goes back to.
    pub configured: ImageDefaults,
    pub companions: Companions,
    /// The adapter the transformer carries, for `/props` — set by `prepare`
    /// after the load, which is where the file was resolved.
    pub adapter: Option<PathBuf>,
    /// One generation at a time — see the module doc.
    busy: Mutex<()>,
    /// Latent-token passes per second, as last measured: seeded from the
    /// startup calibration, replaced by every finished picture's own rate.
    /// What the step-0 progress event and `/props` announce a wait from.
    rate: Mutex<Option<RateModel>>,
}

impl Pipeline {
    // Eight parameters: the three models, the two companions' record, the
    // backend, the defaults, the seed rate and the adapter — each a thing
    // `prepare` resolved separately, and a struct for them would only move
    // the same eight names one line down.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        transformer: &LoadedModel,
        companions: Companions,
        encoder: Arc<dyn ModelForward>,
        tokenizer: Arc<Tokenizer>,
        backend: Arc<dyn Backend>,
        defaults: ImageDefaults,
        rate: Option<RateModel>,
        lora: Option<lora::Lora>,
        merge_lora: bool,
        mut merged_cache: Option<lora::MergedCache>,
        vae_precision: vae::VaePrecision,
    ) -> Result<Self> {
        let started = Instant::now();
        let merging = lora.is_some() && merge_lora;
        let hit = merged_cache.as_ref().is_some_and(|c| c.is_hit());
        let transformer = QwenImageTransformer::load(
            transformer,
            backend.clone(),
            lora,
            merge_lora,
            merged_cache.as_mut(),
        )
        .context("building the qwen_image transformer")?;
        if merging {
            if hit {
                log::info!(
                    "orangu-server: [image] LoRA merge read from {} in {:.0} s",
                    merged_cache
                        .as_ref()
                        .map(|c| c.path().display().to_string())
                        .unwrap_or_default(),
                    started.elapsed().as_secs_f64()
                );
            } else {
                log::info!(
                    "orangu-server: [image] LoRA merged into the transformer's weights in {:.0} s",
                    started.elapsed().as_secs_f64()
                );
                if let Some(cache) = merged_cache.as_mut() {
                    let writing = Instant::now();
                    match cache.flush() {
                        Ok(bytes) if bytes > 0 => log::info!(
                            "orangu-server: [image] merged weights cached at {} ({}, {:.0} s) — \
                             the next start skips the merge",
                            cache.path().display(),
                            orangu::format::format_bytes(bytes),
                            writing.elapsed().as_secs_f64()
                        ),
                        Ok(_) => {}
                        Err(err) => log::warn!(
                            "orangu-server: [image] the merged weights could not be cached at {}: \
                             {err:#} — every start will merge again",
                            cache.path().display()
                        ),
                    }
                }
            }
        }
        ensure!(
            transformer.config.txt_dim as u64 == TEXT_ENCODER_WIDTH
                && encoder.config().n_embd as u64 == TEXT_ENCODER_WIDTH,
            "the transformer reads {}-wide text states but the text encoder produces {}",
            transformer.config.txt_dim,
            encoder.config().n_embd
        );
        let vae = QwenImageVae::load(&companions.vae, backend, vae_precision)
            .context("loading the VAE")?;
        Ok(Self {
            transformer,
            vae,
            encoder,
            tokenizer,
            configured: defaults.clone(),
            defaults: RwLock::new(defaults),
            companions,
            adapter: None,
            busy: Mutex::new(()),
            rate: Mutex::new(rate),
        })
    }

    pub fn transformer_config(&self) -> &transformer::TransformerConfig {
        &self.transformer.config
    }

    /// The current picture defaults.
    pub fn defaults(&self) -> ImageDefaults {
        self.defaults
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Replaces the picture defaults — `POST /props`. Validation is the
    /// caller's (`http::images::apply_settings`): what arrives here is a
    /// set a request could be built from.
    pub fn set_defaults(&self, defaults: ImageDefaults) {
        *self
            .defaults
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = defaults;
    }

    /// A prompt's conditioning: the text encoder's final hidden states for
    /// the prompt's own tokens, `[n, 3584]`.
    fn encode_prompt(&self, prompt: &str) -> Result<Vec<f32>> {
        let prefix = self.tokenizer.encode(PROMPT_TEMPLATE_PREFIX, false);
        let text = format!("{PROMPT_TEMPLATE_PREFIX}{prompt}{PROMPT_TEMPLATE_SUFFIX}");
        let mut tokens = self.tokenizer.encode(&text, false);
        ensure!(
            tokens.starts_with(&prefix),
            "the prompt template did not tokenize as a prefix of the prompt"
        );
        // diffusers truncates the whole rendered text to `max_length +
        // drop_idx`, so the suffix goes first on a long prompt.
        tokens.truncate(prefix.len() + MAX_PROMPT_TOKENS);
        let width = self.encoder.config().n_embd;
        let hidden = self
            .encoder
            .forward_hidden_states(&tokens)
            .context("running the text encoder")?;
        ensure!(
            hidden.len() == tokens.len() * width,
            "text encoder returned the wrong shape"
        );
        let mut out = hidden;
        out.drain(..prefix.len() * width);
        ensure!(!out.is_empty(), "the prompt encoded to no tokens");
        Ok(out)
    }

    /// Runs one request to a finished picture, reporting after every step
    /// and stopping early (with an error) once `cancel` is set.
    pub fn generate(
        &self,
        request: &ImageRequest,
        progress: &mut dyn FnMut(Progress),
        cancel: &AtomicBool,
    ) -> Result<GeneratedImage> {
        let _turn = self
            .busy
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let started = Instant::now();
        let unit = 2 * SPATIAL_COMPRESSION;
        ensure!(
            request.width >= unit
                && request.height >= unit
                && request.width.is_multiple_of(unit)
                && request.height.is_multiple_of(unit),
            "image size must be a multiple of {unit} pixels on each side (got {}x{})",
            request.width,
            request.height
        );
        ensure!(request.steps > 0, "steps must be at least 1");
        ensure!(!request.prompt.trim().is_empty(), "the prompt is empty");
        let (h8, w8) = (
            request.height / SPATIAL_COMPRESSION,
            request.width / SPATIAL_COMPRESSION,
        );
        let grid = (h8 / 2, w8 / 2);
        let n_tokens = grid.0 * grid.1;
        let seed = request.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0)
        });

        let mut timings = Timings::default();
        // The wait, before any of it is spent: a step-0 progress event
        // carrying the per-step estimate from the last measured rate, so a
        // console can show "about 40 minutes" the moment the prompt is sent
        // rather than after the first step (which at 1024 pixels is itself
        // minutes away). `steps` here is the requested count; image-to-image
        // trims it below, and the real events from the loop take over.
        let passes_per_step = if request.cfg_scale > 1.0 { 2.0 } else { 1.0 };
        if let Some(seconds_per_step) = self.seconds_per_step(n_tokens, passes_per_step) {
            progress(Progress {
                step: 0,
                steps: request.steps,
                seconds_per_step,
            });
        }
        // Whatever a previous, cancelled pass left behind is not this
        // picture's.
        let _ = self.transformer.take_stages();
        let positive = self.encode_prompt(&request.prompt)?;
        let guided = request.cfg_scale > 1.0;
        let negative = if guided {
            let text = request
                .negative_prompt
                .clone()
                .unwrap_or_else(|| self.defaults().negative_prompt);
            // An empty negative prompt still encodes to something: the
            // template frames it. A single space is what Qwen's own examples
            // pass, so a blank stays a blank rather than an error.
            let text = if text.is_empty() {
                " ".to_string()
            } else {
                text
            };
            Some(self.encode_prompt(&text)?)
        } else {
            None
        };
        if cancelled(Some(cancel)) {
            bail!("cancelled");
        }

        let sigmas = scheduler::sigmas(&ScheduleConfig::QWEN_IMAGE, request.steps, n_tokens);
        let mut noise = Noise::seeded(seed);
        let token_width = Z_DIM * 4;
        let mut latents = noise.normals(n_tokens * token_width);
        let mut first_step = 0;
        if let Some(init) = &request.init {
            let rgb = codec::decode_to_feature(&init.bytes, request.width, request.height)
                .context("reading the attached picture")?;
            let (mean, logvar) = self
                .vae
                .encode_unless(&rgb, Some(cancel))
                .context("encoding the attached picture")?;
            // The posterior's sample, as diffusers draws it.
            let mut z = mean.data;
            for ((v, lv), n) in z
                .iter_mut()
                .zip(&logvar.data)
                .zip(noise.normals(logvar.data.len()))
            {
                *v += (0.5 * lv.clamp(-30.0, 20.0)).exp() * n;
            }
            normalize_latent(&mut z);
            let clean = pack_latent(&z, h8, w8);
            first_step = scheduler::first_step_for_strength(request.steps, init.strength);
            if first_step >= request.steps {
                // Strength 0: nothing to run; the picture comes straight back
                // through the VAE.
                latents = clean;
            } else {
                latents = scheduler::scale_noise(&clean, &latents, sigmas[first_step]);
            }
        }

        let steps_to_run = request.steps - first_step;
        timings.encode = started.elapsed();
        let loop_started = Instant::now();
        for (done, i) in (first_step..request.steps).enumerate() {
            if cancelled(Some(cancel)) {
                bail!("cancelled");
            }
            let sigma = sigmas[i];
            let mut velocity = self.transformer.forward(&ForwardInput {
                img: &latents,
                grid,
                txt: &positive,
                sigma,
                cancel: Some(cancel),
            })?;
            if let Some(negative) = &negative {
                let unguided = self.transformer.forward(&ForwardInput {
                    img: &latents,
                    grid,
                    txt: negative,
                    sigma,
                    cancel: Some(cancel),
                })?;
                guide(&mut velocity, &unguided, request.cfg_scale, token_width);
            }
            scheduler::euler_step(&mut latents, &velocity, sigma, sigmas[i + 1]);
            progress(Progress {
                step: done + 1,
                steps: steps_to_run,
                seconds_per_step: loop_started.elapsed().as_secs_f64() / (done + 1) as f64,
            });
        }

        timings.denoise = loop_started.elapsed();
        timings.stages = self.transformer.take_stages();
        // The measured rates, for the next picture's estimate — the
        // denoising loop split into attention and the rest by the stage
        // account, the decode per pixel, the encode as it was.
        // Only from a picture big enough to be compute-bound: a 16-pixel
        // warmup is one token, whose pass is a read of every weight — a
        // per-token cost 200× the real one, which once extrapolated to a
        // 256-pixel picture as an hour and a half.
        if steps_to_run > 0
            && timings.denoise.as_secs_f64() > 0.0
            && n_tokens >= RATE_MODEL_MIN_TOKENS
        {
            let token_passes = (n_tokens * steps_to_run) as f64 * passes_per_step;
            let attention = timings.stages.attention.as_secs_f64();
            let linear = (timings.denoise.as_secs_f64() - attention).max(0.0);
            *self
                .rate
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(RateModel {
                linear_per_token_pass: linear / token_passes,
                attention_per_token_pass: attention / token_passes,
                attention_tokens: n_tokens as f64,
                decode_per_pixel: timings.decode.as_secs_f64()
                    / (request.width * request.height) as f64,
                encode: timings.encode.as_secs_f64(),
            });
        }

        let decode_started = Instant::now();
        let mut z = unpack_latent(&latents, h8, w8);
        denormalize_latent(&mut z);
        let rgb = self
            .vae
            .decode(&Feature::new(h8, w8, Z_DIM, z))
            .context("decoding the picture")?;
        let bytes = codec::encode(&rgb, request.format)?;
        timings.decode = decode_started.elapsed();
        log::info!(
            "orangu-server: [image] {}x{} in {:.1}s: encode {:.1}s, {} step(s) {:.1}s ({:.1}s each), \
             decode {:.1}s",
            request.width,
            request.height,
            started.elapsed().as_secs_f64(),
            timings.encode.as_secs_f64(),
            steps_to_run,
            timings.denoise.as_secs_f64(),
            timings.denoise.as_secs_f64() / steps_to_run.max(1) as f64,
            timings.decode.as_secs_f64(),
        );
        let st = &timings.stages;
        let pct =
            |d: std::time::Duration| 100.0 * d.as_secs_f64() / st.total().as_secs_f64().max(1e-9);
        log::info!(
            "orangu-server: [image] {} transformer pass(es): modulation {:.0}%, qkv {:.0}%, \
             attention {:.0}%, out {:.0}%, mlp {:.0}%, other {:.0}%, lora {:.0}%",
            st.passes,
            pct(st.modulation),
            pct(st.qkv),
            pct(st.attention),
            pct(st.out),
            pct(st.mlp),
            pct(st.other),
            pct(st.lora),
        );
        Ok(GeneratedImage {
            bytes,
            format: request.format,
            width: request.width,
            height: request.height,
            seed,
            steps: steps_to_run,
            elapsed: started.elapsed(),
            timings,
        })
    }
}

/// What a background generation reports, in order: progress after every
/// step, then exactly one of `Done` or `Error`.
#[derive(Debug)]
pub enum ImageEvent {
    Progress(Progress),
    /// Boxed: the picture carries its bytes and its timings, and every
    /// progress event would otherwise be sized for them.
    Done(Box<GeneratedImage>),
    Error(String),
}

/// Set once the server is shutting down: every picture in flight stops at
/// its next block or VAE layer — seconds — rather than at its next step,
/// which at 1024 px is minutes the exit would otherwise wait for (the
/// runtime joins the blocking thread a picture runs on).
static SHUTDOWN: AtomicBool = AtomicBool::new(false);

/// Tells every running and future picture to stop.
pub fn request_shutdown() {
    SHUTDOWN.store(true, Ordering::Relaxed);
}

/// Whether the request's own flag, or the server's, says stop.
pub fn cancelled(cancel: Option<&AtomicBool>) -> bool {
    SHUTDOWN.load(Ordering::Relaxed) || cancel.is_some_and(|flag| flag.load(Ordering::Relaxed))
}

/// Stops a background generation when dropped — held by whoever is reading
/// its events, so a client that disconnects mid-picture stops the work at
/// the next step rather than running the remaining minutes for nobody.
pub struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

impl Pipeline {
    /// Runs `request` on a blocking thread, streaming [`ImageEvent`]s back.
    /// Both HTTP endpoints and the web console read the same channel.
    pub fn spawn(
        self: &Arc<Self>,
        request: ImageRequest,
    ) -> (tokio::sync::mpsc::Receiver<ImageEvent>, CancelOnDrop) {
        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let cancel = Arc::new(AtomicBool::new(false));
        let pipeline = self.clone();
        let flag = cancel.clone();
        tokio::task::spawn_blocking(move || {
            let progress_tx = tx.clone();
            let mut progress = |p: Progress| {
                // A full channel means the reader is slower than the steps;
                // dropping a progress frame is fine, the next one supersedes it.
                let _ = progress_tx.try_send(ImageEvent::Progress(p));
            };
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pipeline.generate(&request, &mut progress, &flag)
            }));
            let event = match result {
                Ok(Ok(image)) => ImageEvent::Done(Box::new(image)),
                Ok(Err(err)) => ImageEvent::Error(format!("{err:#}")),
                Err(_) => ImageEvent::Error(
                    crate::panic_capture::take_last_panic_detail()
                        .unwrap_or_else(|| "image generation panicked".to_string()),
                ),
            };
            let _ = tx.blocking_send(event);
        });
        (rx, CancelOnDrop(cancel))
    }
}

/// Classifier-free guidance with Qwen-Image's norm rescale: the guided
/// velocity `neg + scale * (pos - neg)` is scaled back, per token, to the
/// length of the positive prediction.
fn guide(positive: &mut [f32], negative: &[f32], scale: f32, token_width: usize) {
    for (pos, neg) in positive
        .chunks_mut(token_width)
        .zip(negative.chunks(token_width))
    {
        let cond_norm = pos.iter().map(|v| v * v).sum::<f32>().sqrt();
        for (p, n) in pos.iter_mut().zip(neg) {
            *p = n + scale * (*p - n);
        }
        let noise_norm = pos.iter().map(|v| v * v).sum::<f32>().sqrt();
        if noise_norm > 0.0 {
            let k = cond_norm / noise_norm;
            for p in pos.iter_mut() {
                *p *= k;
            }
        }
    }
}

/// `(z - mean) / std`, per channel, on a channel-last latent.
fn normalize_latent(z: &mut [f32]) {
    for px in z.chunks_mut(Z_DIM) {
        for ((v, m), s) in px.iter_mut().zip(&LATENTS_MEAN).zip(&LATENTS_STD) {
            *v = (*v - m) / s;
        }
    }
}

fn denormalize_latent(z: &mut [f32]) {
    for px in z.chunks_mut(Z_DIM) {
        for ((v, m), s) in px.iter_mut().zip(&LATENTS_MEAN).zip(&LATENTS_STD) {
            *v = *v * s + m;
        }
    }
}

/// diffusers' `_pack_latents`: a channel-last `[h8 * w8, 16]` latent into
/// `[(h8/2) * (w8/2), 64]` tokens, each a 2x2 patch with features ordered
/// `(channel, dy, dx)`.
pub(crate) fn pack_latent(z: &[f32], h8: usize, w8: usize) -> Vec<f32> {
    debug_assert_eq!(z.len(), h8 * w8 * Z_DIM);
    let (rows, cols) = (h8 / 2, w8 / 2);
    let mut out = vec![0.0f32; rows * cols * Z_DIM * 4];
    for r in 0..rows {
        for q in 0..cols {
            let token = &mut out[(r * cols + q) * Z_DIM * 4..(r * cols + q + 1) * Z_DIM * 4];
            for c in 0..Z_DIM {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let px = (2 * r + dy) * w8 + (2 * q + dx);
                        token[c * 4 + dy * 2 + dx] = z[px * Z_DIM + c];
                    }
                }
            }
        }
    }
    out
}

/// The inverse of [`pack_latent`].
pub(crate) fn unpack_latent(tokens: &[f32], h8: usize, w8: usize) -> Vec<f32> {
    let (rows, cols) = (h8 / 2, w8 / 2);
    debug_assert_eq!(tokens.len(), rows * cols * Z_DIM * 4);
    let mut z = vec![0.0f32; h8 * w8 * Z_DIM];
    for r in 0..rows {
        for q in 0..cols {
            let token = &tokens[(r * cols + q) * Z_DIM * 4..(r * cols + q + 1) * Z_DIM * 4];
            for c in 0..Z_DIM {
                for dy in 0..2 {
                    for dx in 0..2 {
                        let px = (2 * r + dy) * w8 + (2 * q + dx);
                        z[px * Z_DIM + c] = token[c * 4 + dy * 2 + dx];
                    }
                }
            }
        }
    }
    z
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rate model scales attention with the token count and the rest
    /// linearly: a step at four times the tokens of the measurement costs
    /// four times the linears and sixteen times the attention, and the
    /// whole picture adds the encode and a per-pixel decode.
    #[test]
    fn the_rate_model_grows_attention_with_the_token_count() {
        let m = RateModel {
            linear_per_token_pass: 0.02,
            attention_per_token_pass: 0.004,
            attention_tokens: 1024.0,
            decode_per_pixel: 30e-6,
            encode: 2.0,
        };
        let at_1024 = m.seconds_per_step(1024, 1.0);
        assert!((at_1024 - 1024.0 * 0.024).abs() < 1e-9);
        let at_4096 = m.seconds_per_step(4096, 1.0);
        assert!((at_4096 - 4096.0 * (0.02 + 0.016)).abs() < 1e-6);
        assert!(at_4096 > 4.0 * at_1024);
        // Guidance doubles the passes, and the picture adds encode and decode.
        let picture = m.seconds_for(1024, 1024, 8, 2.0);
        assert!((picture - (2.0 + 8.0 * 2.0 * at_4096 + 30e-6 * 1024.0 * 1024.0)).abs() < 1e-3);
        // Seeded from a calibration alone, the shape is the same.
        let seeded = RateModel::from_calibration(40.0);
        assert!((seeded.linear_per_token_pass - 0.025).abs() < 1e-9);
        assert!(seeded.seconds_per_step(256, 1.0) > 256.0 * 0.025);
    }

    #[test]
    fn packing_is_a_bijection_with_the_diffusers_feature_order() {
        let (h8, w8) = (4, 6);
        let z: Vec<f32> = (0..h8 * w8 * Z_DIM).map(|i| i as f32).collect();
        let tokens = pack_latent(&z, h8, w8);
        assert_eq!(tokens.len(), (h8 / 2) * (w8 / 2) * 64);
        // Token (0,0), channel c=1, dy=1, dx=0 — feature c*4 + dy*2 + dx —
        // is pixel (y=1, x=0) channel 1.
        let (c, dy, dx, y, x) = (1, 1, 0, 1, 0);
        assert_eq!(tokens[c * 4 + dy * 2 + dx], z[(y * w8 + x) * Z_DIM + c]);
        // Token (row 1, col 2) of the 2x3 grid sits at pixels (2..4, 4..6).
        let (row, col, cols) = (1, 2, 3);
        let t = &tokens[(row * cols + col) * 64..(row * cols + col + 1) * 64];
        assert_eq!(t[5 * 4 + 3], z[(3 * w8 + 5) * Z_DIM + 5]);
        assert_eq!(unpack_latent(&tokens, h8, w8), z);
    }

    #[test]
    fn guidance_keeps_the_positive_predictions_length() {
        let mut pos = vec![3.0f32, 4.0, 0.0, 0.0];
        let neg = vec![0.0f32, 0.0, 0.0, 0.0];
        guide(&mut pos, &neg, 4.0, 4);
        // 4x the positive, scaled back to length 5: the same vector.
        assert!((pos[0] - 3.0).abs() < 1e-5 && (pos[1] - 4.0).abs() < 1e-5);
        let mut pos = vec![1.0f32, 0.0];
        let neg = vec![0.0f32, 1.0];
        guide(&mut pos, &neg, 2.0, 2);
        // neg + 2 (pos - neg) = (2, -1), length sqrt(5), scaled to 1.
        let len = (pos[0] * pos[0] + pos[1] * pos[1]).sqrt();
        assert!((len - 1.0).abs() < 1e-5);
        assert!(pos[0] > 0.0 && pos[1] < 0.0);
    }

    #[test]
    fn latent_normalisation_round_trips() {
        let mut z: Vec<f32> = (0..Z_DIM * 2).map(|i| i as f32 * 0.25 - 1.0).collect();
        let original = z.clone();
        normalize_latent(&mut z);
        assert!((z[0] - (original[0] - LATENTS_MEAN[0]) / LATENTS_STD[0]).abs() < 1e-6);
        denormalize_latent(&mut z);
        for (a, b) in z.iter().zip(&original) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    /// One small picture end to end on the CPU: prompt in, PNG out. Slow
    /// (minutes: two transformer passes over 20B parameters at 256 tokens)
    /// and needing three real files, so ignored by default.
    ///
    /// Run with `ORANGU_TEST_QWEN_IMAGE_MODEL=/path/to/qwen-image-2512-Q4_K_M.gguf
    /// cargo test --release --bin orangu-server a_small_picture -- --ignored`;
    /// the encoder and VAE are found under `ORANGU_TEST_QWEN_IMAGE_MODELS_DIR`,
    /// or the hub-cache root the model sits in.
    #[test]
    #[ignore]
    fn a_small_picture_is_generated_end_to_end_on_the_cpu() {
        let model = std::env::var("ORANGU_TEST_QWEN_IMAGE_MODEL")
            .expect("set ORANGU_TEST_QWEN_IMAGE_MODEL");
        let model = PathBuf::from(model);
        let models_dir = std::env::var("ORANGU_TEST_QWEN_IMAGE_MODELS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                // `<models>/models--x--y/snapshots/<commit>/<file>`.
                model
                    .ancestors()
                    .nth(4)
                    .expect("the model sits in a hub-cache layout")
                    .to_path_buf()
            });
        let companions = Companions::locate(&models_dir, None, None).expect("companions");
        let backend: Arc<dyn Backend> = Arc::new(crate::engine::backend::CpuBackend);
        let encoder_loaded = LoadedModel::open(&companions.text_encoder).expect("encoder");
        let encoder: Arc<dyn ModelForward> = Arc::new(
            crate::engine::arch::llama::LlamaModel::load_with_backend(
                &encoder_loaded,
                backend.clone(),
            )
            .expect("build encoder"),
        );
        let gguf = orangu::gguf::GgufFile::open(&companions.text_encoder).expect("encoder gguf");
        let tokenizer = Arc::new(Tokenizer::from_gguf(&gguf).expect("tokenizer"));
        let transformer = LoadedModel::open(&model).expect("transformer");
        let pipeline = Pipeline::load(
            &transformer,
            companions,
            encoder,
            tokenizer,
            backend,
            ImageDefaults::default(),
            None,
            None,
            true,
            None,
            vae::VaePrecision::default(),
        )
        .expect("pipeline");

        let mut steps_seen = 0;
        let image = pipeline
            .generate(
                &ImageRequest {
                    prompt: "a red circle on a white background".into(),
                    negative_prompt: None,
                    width: 256,
                    height: 256,
                    steps: 2,
                    cfg_scale: 1.0,
                    seed: Some(1),
                    format: ImageFormat::Png,
                    init: None,
                },
                &mut |p| {
                    steps_seen = p.step;
                    eprintln!(
                        "step {}/{} ({:.1}s per step)",
                        p.step, p.steps, p.seconds_per_step
                    );
                },
                &AtomicBool::new(false),
            )
            .expect("generate");
        assert_eq!(steps_seen, 2);
        assert_eq!((image.width, image.height), (256, 256));
        assert_eq!(codec::dimensions(&image.bytes).unwrap(), (256, 256));
        let out = std::env::temp_dir().join("orangu-qwen-image-test.png");
        std::fs::write(&out, &image.bytes).unwrap();
        eprintln!("wrote {} in {:?}", out.display(), image.elapsed);
        // A generated picture is not a constant: every channel must vary.
        let rgb = codec::decode_to_feature(&image.bytes, 256, 256).unwrap();
        for c in 0..3 {
            let values: Vec<f32> = rgb.data.iter().skip(c).step_by(3).copied().collect();
            let min = values.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            assert!(max - min > 0.05, "channel {c} is flat ({min}..{max})");
        }
    }

    #[test]
    fn the_prompt_template_is_diffusers_own() {
        // The exact string, since the encoder's hidden states depend on it
        // and a changed word would silently condition every picture on
        // different context.
        assert!(PROMPT_TEMPLATE_PREFIX.starts_with("<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"));
        assert_eq!(
            PROMPT_TEMPLATE_SUFFIX,
            "<|im_end|>\n<|im_start|>assistant\n"
        );
    }
}
