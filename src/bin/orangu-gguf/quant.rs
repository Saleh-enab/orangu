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

//! Weights to bytes: the float encodings, the block quantizations, and the
//! per-tensor type rules that turn a file type like `Q4_K_M` into an actual
//! type for each tensor.
//!
//! A `Q4_K_M` file is not a file of `Q4_K` tensors. The M is a *mixture*:
//! the tensors that a quantization error hurts most — the vocabulary
//! projection, the value projections and the down projections in the outer
//! blocks — are carried at higher precision, and the rest at four bits.
//! The rules here are the established ones, spelled out so a file this tool
//! writes is the same mixture a reader expects when it sees that name.
//!
//! Two properties matter more than the compression ratio:
//!
//! - **Norms and any other 1-D tensor stay `f32`.** They are a rounding
//!   error's worth of file size and the thing every activation is divided
//!   by.
//! - **A block quantization needs its row length to divide the block.**
//!   A row that does not divide 256 cannot be a K-quant at all, so it falls
//!   back — to a 32-wide block where one exists, and to `f16` when even
//!   that does not fit. Falling back silently would be the wrong thing to
//!   do, so [`Plan`] records every fallback for the caller to report.

use anyhow::{Result, bail};
use half::f16;

// The types this tool writes, and the ones it only has to name. The
// round-number quantizations — `Q4_0`, `Q5_0`, `Q8_0` and the rest — are
// deliberately absent: they are the pre-K-quant generation, a K-quant beats
// them at every size, and a writer that offers both mostly offers a way to
// pick the worse one.
pub use orangu::quantize::*;

/// A whole-file type: what `--quantization` selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ftype {
    F32,
    F16,
    Bf16,
    Q2K,
    Q3KS,
    Q3KM,
    Q3KL,
    Q4KS,
    Q4KM,
    Q5KS,
    Q5KM,
    Q6K,
    IQ4NL,
    IQ4XS,
}

/// The file types that exist in the format but that this tool will not
/// write, and the reason, so `--quantization iq2_xs` says something more
/// useful than "unknown".
///
/// Every one of them is a search against a fixed codebook of lattice
/// points, and below about three bits that search only lands anywhere
/// useful when it is told which weights matter — an importance matrix,
/// measured by running the model over calibration text. Without one the
/// reference implementation refuses outright rather than write a file that
/// looks fine and answers badly. This tool has no importance matrix pass
/// yet, so it refuses for the same reason.
const NEEDS_IMPORTANCE: [&str; 10] = [
    "q2_k_s", "iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s", "iq2_m", "iq3_xxs", "iq3_s", "iq3_m",
];

impl Ftype {
    /// Parses the spelling used on the command line.
    pub fn parse(value: &str) -> Result<Self> {
        let spelling = value.trim().to_ascii_lowercase();
        Ok(match spelling.as_str() {
            "f32" | "fp32" => Ftype::F32,
            "f16" | "fp16" => Ftype::F16,
            "bf16" | "bfloat16" => Ftype::Bf16,
            "q2_k" | "q2k" => Ftype::Q2K,
            "q3_k_s" | "q3ks" => Ftype::Q3KS,
            "q3_k_m" | "q3km" | "q3_k" => Ftype::Q3KM,
            "q3_k_l" | "q3kl" => Ftype::Q3KL,
            "q4_k_s" | "q4ks" => Ftype::Q4KS,
            "q4_k_m" | "q4km" | "q4_k" => Ftype::Q4KM,
            "q5_k_s" | "q5ks" => Ftype::Q5KS,
            "q5_k_m" | "q5km" | "q5_k" => Ftype::Q5KM,
            "q6_k" | "q6k" => Ftype::Q6K,
            "iq4_nl" | "iq4nl" => Ftype::IQ4NL,
            "iq4_xs" | "iq4xs" => Ftype::IQ4XS,
            other if NEEDS_IMPORTANCE.contains(&other) => bail!(
                "{} needs an importance matrix, which this tool does not measure yet — \
                 at that many bits a file written without one is worse than the next size up",
                other.to_ascii_uppercase()
            ),
            other
                if other.starts_with("q4_0")
                    || other.starts_with("q4_1")
                    || other.starts_with("q5_0")
                    || other.starts_with("q5_1")
                    || other.starts_with("q8_0")
                    || other.starts_with("q8_1") =>
            {
                bail!(
                    "{} is a pre-K-quant format and is not written — Q6_K is the same size and more accurate",
                    other.to_ascii_uppercase()
                )
            }
            other => bail!(
                "unknown quantization {other:?} — one of: {}",
                Ftype::ALL
                    .iter()
                    .map(|f| f.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        })
    }

    /// Largest first, so a list of them reads as a size ladder.
    pub const ALL: [Ftype; 14] = [
        Ftype::Bf16,
        Ftype::F16,
        Ftype::F32,
        Ftype::Q6K,
        Ftype::Q5KM,
        Ftype::Q5KS,
        Ftype::Q4KM,
        Ftype::Q4KS,
        Ftype::IQ4NL,
        Ftype::IQ4XS,
        Ftype::Q3KL,
        Ftype::Q3KM,
        Ftype::Q3KS,
        Ftype::Q2K,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            Ftype::F32 => "F32",
            Ftype::F16 => "F16",
            Ftype::Bf16 => "BF16",
            Ftype::Q2K => "Q2_K",
            Ftype::Q3KS => "Q3_K_S",
            Ftype::Q3KM => "Q3_K_M",
            Ftype::Q3KL => "Q3_K_L",
            Ftype::Q4KS => "Q4_K_S",
            Ftype::Q4KM => "Q4_K_M",
            Ftype::Q5KS => "Q5_K_S",
            Ftype::Q5KM => "Q5_K_M",
            Ftype::Q6K => "Q6_K",
            Ftype::IQ4NL => "IQ4_NL",
            Ftype::IQ4XS => "IQ4_XS",
        }
    }

    /// One line on what this file type is for.
    pub fn description(&self) -> &'static str {
        match self {
            Ftype::F32 => {
                "Every weight in full precision. Twice the size of BF16, no more accurate in practice."
            }
            Ftype::F16 => {
                "Half precision. Same size as BF16, less range — BF16 is the better default for trained weights."
            }
            Ftype::Bf16 => {
                "Half precision with the exponent range of F32. The reference file: quantize from this, not to it."
            }
            Ftype::Q2K => {
                "Two bits. The smallest file worth writing, and the first one where the model is visibly worse."
            }
            Ftype::Q3KS => "Three bits throughout. Smaller than Q3_K_M and measurably weaker.",
            Ftype::Q3KM => {
                "Three bits, with four and five where they pay: the value and down projections."
            }
            Ftype::Q3KL => "Three bits, with five on the tensors Q3_K_M gives four.",
            Ftype::Q4KS => "Four bits, with five on the outermost down projections only.",
            Ftype::Q4KM => {
                "Four bits, with six on the outer value and down projections. The size/quality sweet spot."
            }
            Ftype::Q5KS => "Five bits throughout. Close to Q6_K for noticeably less size.",
            Ftype::Q5KM => "Five bits, with six on the outer value and down projections.",
            Ftype::Q6K => {
                "Six bits. Indistinguishable from BF16 on every benchmark that matters, at 40% of the size."
            }
            Ftype::IQ4NL => {
                "Four non-linear bits in 32-wide blocks. Fits rows a K-quant cannot, at Q4_K_S's size."
            }
            Ftype::IQ4XS => {
                "Four non-linear bits in 256-wide blocks. The smallest four-bit file, just under Q4_K_S."
            }
        }
    }

    /// The `general.file_type` this writes.
    pub fn file_type(&self) -> u32 {
        match self {
            Ftype::F32 => 0,
            Ftype::F16 => 1,
            Ftype::Q2K => 10,
            Ftype::Q3KS => 11,
            Ftype::Q3KM => 12,
            Ftype::Q3KL => 13,
            Ftype::Q4KS => 14,
            Ftype::Q4KM => 15,
            Ftype::Q5KS => 16,
            Ftype::Q5KM => 17,
            Ftype::Q6K => 18,
            Ftype::IQ4NL => 25,
            Ftype::IQ4XS => 30,
            Ftype::Bf16 => 32,
        }
    }

    /// The type most of the file is written as, before the mixture rules
    /// move individual tensors.
    fn base_type(&self) -> u32 {
        match self {
            Ftype::F32 => GGML_TYPE_F32,
            Ftype::F16 => GGML_TYPE_F16,
            Ftype::Bf16 => GGML_TYPE_BF16,
            Ftype::Q2K => GGML_TYPE_Q2_K,
            Ftype::Q3KS | Ftype::Q3KM | Ftype::Q3KL => GGML_TYPE_Q3_K,
            Ftype::Q4KS | Ftype::Q4KM => GGML_TYPE_Q4_K,
            Ftype::Q5KS | Ftype::Q5KM => GGML_TYPE_Q5_K,
            Ftype::Q6K => GGML_TYPE_Q6_K,
            Ftype::IQ4NL => GGML_TYPE_IQ4_NL,
            Ftype::IQ4XS => GGML_TYPE_IQ4_XS,
        }
    }

    /// Whether the file is quantized at all, which is what decides if the
    /// mixture rules run.
    fn is_quantized(&self) -> bool {
        !matches!(self, Ftype::F32 | Ftype::F16 | Ftype::Bf16)
    }
}

/// Where a tensor sits in the network, which is what the mixture rules key
/// off. Derived from the tensor's name so it works for a file this tool
/// wrote and for one it is only reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    TokenEmbedding,
    Output,
    AttentionValue,
    AttentionOutput,
    FeedForwardDown,
    Other,
}

pub fn role_of(name: &str) -> Role {
    if name == "token_embd.weight" {
        Role::TokenEmbedding
    } else if name == "output.weight" {
        Role::Output
    } else if name.ends_with("attn_v.weight") {
        Role::AttentionValue
    } else if name.ends_with("attn_output.weight") {
        Role::AttentionOutput
    } else if name.ends_with("ffn_down.weight") {
        Role::FeedForwardDown
    } else {
        Role::Other
    }
}

/// The block index in a `blk.N.` tensor name.
pub fn block_index(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

/// The rule that spends the extra bits on the outer blocks and on every
/// third block in between, rather than spreading them evenly.
fn use_more_bits(layer: usize, layers: usize) -> bool {
    layer < layers / 8 || layer >= 7 * layers / 8 || (layer - layers / 8) % 3 == 2
}

/// What the mixture rules need to know about the model beyond one tensor's
/// own name and shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Model {
    pub layers: usize,
    /// Query heads per key/value head. Grouped-query attention makes the
    /// value projection small enough that carrying it at a higher precision
    /// costs almost nothing, so the rules spend bits there when it is 4 or
    /// more.
    pub gqa: usize,
}

/// What one tensor is written as, and why it might not be what was asked
/// for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Plan {
    pub ggml_type: u32,
    /// The type the mixture rules chose, when a shape forced something
    /// else. `None` when nothing was overridden.
    pub fallback_from: Option<u32>,
}

/// Chooses the type for one tensor.
///
/// `dims` is in GGUF order, so `dims[0]` is the row length — the dimension
/// a block has to divide.
pub fn plan_tensor(ftype: Ftype, name: &str, dims: &[u64], model: Model) -> Plan {
    // Anything that is not a matrix — every norm — stays f32, in every
    // file type. They are the divisor of every activation and a rounding
    // error's worth of file size, and a reader expects to find them exact.
    if dims.len() < 2 {
        return Plan {
            ggml_type: GGML_TYPE_F32,
            fallback_from: None,
        };
    }

    let wanted = if ftype.is_quantized() {
        mixture(ftype, name, model)
    } else {
        ftype.base_type()
    };

    let ncols = dims[0] as usize;
    let ggml_type = fit(wanted, ncols);
    Plan {
        ggml_type,
        fallback_from: (ggml_type != wanted).then_some(wanted),
    }
}

/// The per-tensor rules behind the `_S`/`_M`/`_L` in a file type's name.
///
/// A `Q4_K_M` file is not a file of `Q4_K` tensors, and the difference
/// between the three suffixes is entirely here: which tensors are carried
/// above the file's base type, and by how much. The tensors that earn it
/// are always the same ones — the vocabulary projection, because it is the
/// last thing before the softmax; the value and down projections, because
/// an error there lands on every token that attends through them.
fn mixture(ftype: Ftype, name: &str, model: Model) -> u32 {
    let base = ftype.base_type();
    let layers = model.layers.max(1);
    let layer = block_index(name).unwrap_or(0);

    match role_of(name) {
        // The vocabulary projection: six bits in every quantized file.
        Role::Output => GGML_TYPE_Q6_K,
        Role::TokenEmbedding => base,
        Role::AttentionValue => match ftype {
            Ftype::Q2K if model.gqa >= 4 => GGML_TYPE_Q4_K,
            Ftype::Q2K => GGML_TYPE_Q3_K,
            Ftype::Q3KM if layer < 2 => GGML_TYPE_Q5_K,
            Ftype::Q3KM => GGML_TYPE_Q4_K,
            Ftype::Q3KL => GGML_TYPE_Q5_K,
            Ftype::IQ4NL | Ftype::IQ4XS if model.gqa >= 4 => GGML_TYPE_Q5_K,
            Ftype::Q4KM | Ftype::Q5KM if use_more_bits(layer, layers) => GGML_TYPE_Q6_K,
            Ftype::Q4KS if layer < 4 => GGML_TYPE_Q5_K,
            _ => base,
        },
        Role::FeedForwardDown => match ftype {
            Ftype::Q2K => GGML_TYPE_Q3_K,
            Ftype::Q3KM if layer < layers / 16 => GGML_TYPE_Q5_K,
            Ftype::Q3KM => GGML_TYPE_Q4_K,
            Ftype::Q3KL => GGML_TYPE_Q5_K,
            Ftype::Q4KM | Ftype::Q5KM if use_more_bits(layer, layers) => GGML_TYPE_Q6_K,
            Ftype::Q4KS | Ftype::IQ4NL | Ftype::IQ4XS if layer < layers / 8 => GGML_TYPE_Q5_K,
            _ => base,
        },
        Role::AttentionOutput => match ftype {
            Ftype::Q2K => GGML_TYPE_Q3_K,
            Ftype::Q3KM => GGML_TYPE_Q4_K,
            Ftype::Q3KL => GGML_TYPE_Q5_K,
            _ => base,
        },
        Role::Other => base,
    }
}

/// Demotes a type whose block does not divide the row length.
///
/// A 256-wide K-quant that does not fit has one place left to go: `IQ4_NL`
/// is the only 32-wide type here, so anything at four bits or below lands
/// there. Five and six bits do not — dropping them to four to save a fifth
/// of a tensor is the wrong trade on the two tensors that are carried high
/// precisely because they matter — so those fall to `f16`, which needs no
/// block at all.
fn fit(wanted: u32, ncols: usize) -> u32 {
    if ncols.is_multiple_of(block_size(wanted)) {
        return wanted;
    }
    let demoted = match wanted {
        GGML_TYPE_Q2_K | GGML_TYPE_Q3_K | GGML_TYPE_Q4_K | GGML_TYPE_IQ4_XS => GGML_TYPE_IQ4_NL,
        other => other,
    };
    if ncols.is_multiple_of(block_size(demoted)) {
        demoted
    } else {
        GGML_TYPE_F16
    }
}

/// Reads a tensor's stored bytes back as `f32`.
///
/// Only the float encodings are handled, and that is deliberate: this tool
/// quantizes *from* a full-precision file. Quantizing a file that is
/// already quantized would compound two roundings and quietly produce
/// something worse than quantizing the original once, so it is refused with
/// a message that says so.
pub fn decode(ggml_type: u32, bytes: &[u8], elements: usize) -> Result<Vec<f32>> {
    match ggml_type {
        GGML_TYPE_F32 => Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .take(elements)
            .map(|b| f32::from_le_bytes(*b))
            .collect()),
        GGML_TYPE_F16 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .take(elements)
            .map(|b| f16::from_le_bytes(*b).to_f32())
            .collect()),
        GGML_TYPE_BF16 => Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .take(elements)
            .map(|b| f32::from_bits((u16::from_le_bytes(*b) as u32) << 16))
            .collect()),
        other => bail!(
            "this file's tensors are already {} — quantize from the F32, F16 or BF16 file instead",
            type_name(other)
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decoding is the reader's half of the contract, written out here so
    /// the round-trip tests measure the real error rather than this file's
    /// own idea of it.
    fn dequantize(ggml_type: u32, bytes: &[u8], elements: usize) -> Vec<f32> {
        match ggml_type {
            GGML_TYPE_Q2_K => bytes
                .as_chunks::<{ QK_K / 16 + QK_K / 4 + 4 }>()
                .0
                .iter()
                .flat_map(|block| {
                    let scales = &block[..QK_K / 16];
                    let quants = &block[QK_K / 16..QK_K / 16 + QK_K / 4];
                    let at = QK_K / 16 + QK_K / 4;
                    let d = f16::from_le_bytes([block[at], block[at + 1]]).to_f32();
                    let dmin = f16::from_le_bytes([block[at + 2], block[at + 3]]).to_f32();
                    (0..QK_K).map(move |i| {
                        let chunk = i / 128;
                        let within = i % 128;
                        let q = (quants[chunk * 32 + within % 32] >> (2 * (within / 32))) & 3;
                        let scale = scales[i / 16];
                        d * (scale & 0xF) as f32 * q as f32 - dmin * (scale >> 4) as f32
                    })
                })
                .take(elements)
                .collect(),
            GGML_TYPE_Q3_K => bytes
                .as_chunks::<{ QK_K / 8 + QK_K / 4 + 12 + 2 }>()
                .0
                .iter()
                .flat_map(|block| {
                    let mask = &block[..QK_K / 8];
                    let quants = &block[QK_K / 8..QK_K / 8 + QK_K / 4];
                    let mut packed = [0u8; 12];
                    packed.copy_from_slice(&block[QK_K / 8 + QK_K / 4..QK_K / 8 + QK_K / 4 + 12]);
                    let at = QK_K / 8 + QK_K / 4 + 12;
                    let d = f16::from_le_bytes([block[at], block[at + 1]]).to_f32();
                    (0..QK_K).map(move |i| {
                        let chunk = i / 128;
                        let within = i % 128;
                        let low = (quants[chunk * 32 + within % 32] >> (2 * (within / 32))) & 3;
                        let high = (mask[i % (QK_K / 8)] >> (i / (QK_K / 8))) & 1;
                        let q = low as i32 - if high == 0 { 4 } else { 0 };
                        d * unpack_signed_scale(i / 16, &packed) as f32 * q as f32
                    })
                })
                .take(elements)
                .collect(),
            GGML_TYPE_Q5_K => bytes
                .as_chunks::<{ 2 + 2 + 12 + QK_K / 8 + QK_K / 2 }>()
                .0
                .iter()
                .flat_map(|block| {
                    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
                    let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
                    let mut packed = [0u8; 12];
                    packed.copy_from_slice(&block[4..16]);
                    let high = &block[16..16 + QK_K / 8];
                    let low = &block[16 + QK_K / 8..];
                    (0..QK_K).map(move |i| {
                        let j = i / 32;
                        let (sc, m) = unpack_scale_min(j, &packed);
                        let chunk = i / 64;
                        let within = i % 64;
                        let byte = low[chunk * 32 + within % 32];
                        let nibble = if within < 32 { byte & 0xF } else { byte >> 4 };
                        let bit = (high[i % 32] >> (2 * chunk + within / 32)) & 1;
                        let q = nibble | (bit << 4);
                        d * sc as f32 * q as f32 - dmin * m as f32
                    })
                })
                .take(elements)
                .collect(),
            GGML_TYPE_IQ4_NL => bytes
                .as_chunks::<{ 2 + QK / 2 }>()
                .0
                .iter()
                .flat_map(|block| {
                    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
                    let quants = &block[2..];
                    (0..QK).map(move |i| {
                        let byte = quants[i % (QK / 2)];
                        let level = if i < QK / 2 { byte & 0xF } else { byte >> 4 };
                        d * IQ4_VALUES[level as usize] as f32
                    })
                })
                .take(elements)
                .collect(),
            GGML_TYPE_IQ4_XS => bytes
                .as_chunks::<{ 2 + 2 + QK_K / 64 + QK_K / 2 }>()
                .0
                .iter()
                .flat_map(|block| {
                    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
                    let scales_h = u16::from_le_bytes([block[2], block[3]]);
                    let scales_l = &block[4..4 + QK_K / 64];
                    let quants = &block[4 + QK_K / 64..];
                    (0..QK_K).map(move |i| {
                        let ib = i / 32;
                        let low = if ib % 2 == 0 {
                            scales_l[ib / 2] & 0xF
                        } else {
                            scales_l[ib / 2] >> 4
                        };
                        let high = ((scales_h >> (2 * ib)) & 3) as u8;
                        let scale = (low | (high << 4)) as i32 - 32;
                        let within = i % 32;
                        let byte = quants[16 * ib + within % 16];
                        let level = if within < 16 { byte & 0xF } else { byte >> 4 };
                        d * scale as f32 * IQ4_VALUES[level as usize] as f32
                    })
                })
                .take(elements)
                .collect(),
            GGML_TYPE_Q4_K => bytes
                .as_chunks::<{ 2 + 2 + 12 + QK_K / 2 }>()
                .0
                .iter()
                .flat_map(|block| {
                    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
                    let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
                    let mut packed = [0u8; 12];
                    packed.copy_from_slice(&block[4..16]);
                    let quants = &block[16..];
                    (0..QK_K).map(move |i| {
                        let j = i / 32;
                        let (sc, m) = unpack_scale_min(j, &packed);
                        let chunk = i / 64;
                        let within = i % 64;
                        let byte = quants[chunk * 32 + within % 32];
                        let q = if within < 32 { byte & 0xF } else { byte >> 4 };
                        d * sc as f32 * q as f32 - dmin * m as f32
                    })
                })
                .take(elements)
                .collect(),
            GGML_TYPE_Q6_K => bytes
                .as_chunks::<{ QK_K / 2 + QK_K / 4 + QK_K / 16 + 2 }>()
                .0
                .iter()
                .flat_map(|block| {
                    let low = &block[..QK_K / 2];
                    let high = &block[QK_K / 2..QK_K / 2 + QK_K / 4];
                    let scales = &block[QK_K / 2 + QK_K / 4..QK_K / 2 + QK_K / 4 + QK_K / 16];
                    let at = QK_K / 2 + QK_K / 4 + QK_K / 16;
                    let d = f16::from_le_bytes([block[at], block[at + 1]]).to_f32();
                    (0..QK_K).map(move |i| {
                        let chunk = i / 128;
                        let within = i % 128;
                        let quarter = within / 32;
                        let lane = within % 32;
                        let low_byte =
                            low[chunk * 64 + lane + if quarter % 2 == 1 { 32 } else { 0 }];
                        let nibble = if quarter < 2 {
                            low_byte & 0xF
                        } else {
                            low_byte >> 4
                        };
                        let bits = (high[chunk * 32 + lane] >> (2 * quarter)) & 3;
                        let q = (nibble | (bits << 4)) as i32 - 32;
                        let scale = scales[i / 16] as i8;
                        d * scale as f32 * q as f32
                    })
                })
                .take(elements)
                .collect(),
            other => decode(other, bytes, elements).unwrap(),
        }
    }

    /// Weight-shaped values: a deterministic normal sample with the
    /// occasional large outlier, which is what a trained matrix actually
    /// looks like and what the block quantizations are tuned for. A uniform
    /// sample would flatter them; a sawtooth would libel them.
    fn sample(n: usize) -> Vec<f32> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u32 << 24) as f32
        };
        (0..n)
            .map(|i| {
                let u1 = next().max(f32::MIN_POSITIVE);
                let u2 = next();
                let normal = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                normal * 0.02 * if i % 401 == 0 { 5.0 } else { 1.0 }
            })
            .collect()
    }

    fn relative_error(original: &[f32], restored: &[f32]) -> f32 {
        let num: f32 = original
            .iter()
            .zip(restored)
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let den: f32 = original.iter().map(|a| a * a).sum();
        (num / den).sqrt()
    }

    #[test]
    fn bf16_keeps_the_exponent_and_rounds_the_mantissa() {
        assert_eq!(bf16_bytes(1.0), [0x80, 0x3f]);
        assert_eq!(bf16_bytes(-2.0), [0x00, 0xc0]);
        // A value f16 cannot hold at all still round-trips in bf16.
        let big = 1.0e30f32;
        let restored = decode(GGML_TYPE_BF16, &bf16_bytes(big), 1).unwrap()[0];
        assert!((restored / big - 1.0).abs() < 0.01, "{restored}");
    }

    #[test]
    fn every_encoder_round_trips_within_its_precision() {
        let values = sample(2048);
        // Each bound is just above the error the encoder actually
        // achieves on this sample, and each is within a few percent of the
        // theoretical one for that many levels over a Gaussian block — so a
        // regression that costs real precision fails here rather than
        // passing under a bound set loosely enough to hide it.
        for (ggml_type, tolerance) in [
            (GGML_TYPE_F32, 0.0),
            (GGML_TYPE_F16, 3e-4),
            (GGML_TYPE_BF16, 2e-3),
            (GGML_TYPE_Q6_K, 2e-2),
            (GGML_TYPE_Q5_K, 3.8e-2),
            (GGML_TYPE_Q4_K, 7.5e-2),
            (GGML_TYPE_IQ4_XS, 8.2e-2),
            (GGML_TYPE_IQ4_NL, 8.1e-2),
            (GGML_TYPE_Q3_K, 1.6e-1),
            (GGML_TYPE_Q2_K, 3.1e-1),
        ] {
            let bytes = encode(ggml_type, &values, values.len());
            assert_eq!(
                bytes.len(),
                row_bytes(ggml_type, values.len()),
                "{} size",
                type_name(ggml_type)
            );
            let restored = dequantize(ggml_type, &bytes, values.len());
            let error = relative_error(&values, &restored);
            assert!(
                error <= tolerance,
                "{}: relative error {error} over {tolerance}",
                type_name(ggml_type)
            );
        }
    }

    /// More bits must mean less error — a mixture built on types that do
    /// not order this way would be spending size for nothing.
    #[test]
    fn the_types_order_by_accuracy() {
        let values = sample(1024);
        let error = |t: u32| {
            relative_error(
                &values,
                &dequantize(t, &encode(t, &values, values.len()), values.len()),
            )
        };
        assert!(error(GGML_TYPE_Q6_K) < error(GGML_TYPE_Q5_K));
        assert!(error(GGML_TYPE_Q5_K) < error(GGML_TYPE_Q4_K));
        assert!(error(GGML_TYPE_Q4_K) < error(GGML_TYPE_Q3_K));
        assert!(error(GGML_TYPE_Q3_K) < error(GGML_TYPE_Q2_K));
        // The two non-linear four-bit types sit beside the linear one
        // rather than above or below it — that is the whole claim of a
        // codebook, and it is worth pinning.
        assert!(error(GGML_TYPE_IQ4_XS) < error(GGML_TYPE_Q3_K));
        assert!(error(GGML_TYPE_IQ4_NL) < error(GGML_TYPE_Q3_K));
    }

    #[test]
    fn an_all_zero_block_encodes_and_reads_back_as_zero() {
        let values = vec![0.0f32; 512];
        for ggml_type in [
            GGML_TYPE_Q2_K,
            GGML_TYPE_Q3_K,
            GGML_TYPE_Q4_K,
            GGML_TYPE_Q5_K,
            GGML_TYPE_Q6_K,
            GGML_TYPE_IQ4_NL,
            GGML_TYPE_IQ4_XS,
        ] {
            let bytes = encode(ggml_type, &values, values.len());
            let restored = dequantize(ggml_type, &bytes, values.len());
            assert!(
                restored.iter().all(|v| *v == 0.0),
                "{} produced non-zero values from zeros",
                type_name(ggml_type)
            );
        }
    }

    /// A dense model with grouped-query attention: 24 blocks, four query
    /// heads per key/value head.
    const MODEL: Model = Model { layers: 24, gqa: 4 };

    /// The mixture is the whole point of the M in Q4_K_M.
    #[test]
    fn q4_k_m_spends_its_extra_bits_where_the_rules_say() {
        let plan = |name: &str, dims: &[u64]| plan_tensor(Ftype::Q4KM, name, dims, MODEL);

        assert_eq!(
            plan("output.weight", &[2048, 32768]).ggml_type,
            GGML_TYPE_Q6_K
        );
        assert_eq!(
            plan("token_embd.weight", &[2048, 32768]).ggml_type,
            GGML_TYPE_Q4_K
        );
        // The outer blocks get six bits, the middle ones four.
        assert_eq!(
            plan("blk.0.attn_v.weight", &[2048, 1024]).ggml_type,
            GGML_TYPE_Q6_K
        );
        assert_eq!(
            plan("blk.23.ffn_down.weight", &[8192, 2048]).ggml_type,
            GGML_TYPE_Q6_K
        );
        assert_eq!(
            plan("blk.4.attn_v.weight", &[2048, 1024]).ggml_type,
            GGML_TYPE_Q4_K
        );
        assert_eq!(
            plan("blk.4.attn_q.weight", &[2048, 2048]).ggml_type,
            GGML_TYPE_Q4_K
        );
        // Norms are never quantized.
        assert_eq!(
            plan("blk.4.attn_norm.weight", &[2048]).ggml_type,
            GGML_TYPE_F32
        );
        for ftype in Ftype::ALL {
            assert_eq!(
                plan_tensor(ftype, "blk.4.attn_norm.weight", &[2048], MODEL).ggml_type,
                GGML_TYPE_F32,
                "{} quantized a norm",
                ftype.name()
            );
        }
    }

    /// A row that no block divides has to demote, and say that it did.
    #[test]
    fn an_awkward_row_length_falls_back_and_is_recorded() {
        // 896 is not a multiple of 256 but is one of 32, so four bits
        // survive as the non-linear type.
        let plan = plan_tensor(Ftype::Q4KM, "blk.1.attn_q.weight", &[896, 896], MODEL);
        assert_eq!(plan.ggml_type, GGML_TYPE_IQ4_NL);
        assert_eq!(plan.fallback_from, Some(GGML_TYPE_Q4_K));

        // Six bits has nowhere narrower to go that is still six bits, so
        // it goes to f16 rather than halving the precision of the tensor
        // it was chosen for.
        let plan = plan_tensor(Ftype::Q4KM, "output.weight", &[100, 32], MODEL);
        assert_eq!(plan.ggml_type, GGML_TYPE_F16);
        assert_eq!(plan.fallback_from, Some(GGML_TYPE_Q6_K));

        let plan = plan_tensor(Ftype::Q4KM, "blk.1.attn_q.weight", &[2048, 2048], MODEL);
        assert_eq!(plan.fallback_from, None);
    }

    /// The suffix is a promise about which tensors are carried above the
    /// base type. Each of these is the difference between two file types
    /// that would otherwise be the same file.
    #[test]
    fn the_suffixes_differ_where_they_claim_to() {
        let plan = |ftype, name: &str| plan_tensor(ftype, name, &[2048, 2048], MODEL).ggml_type;

        // S carries nothing extra in the middle of the stack; M and L do.
        assert_eq!(plan(Ftype::Q3KS, "blk.12.ffn_down.weight"), GGML_TYPE_Q3_K);
        assert_eq!(plan(Ftype::Q3KM, "blk.12.ffn_down.weight"), GGML_TYPE_Q4_K);
        assert_eq!(plan(Ftype::Q3KL, "blk.12.ffn_down.weight"), GGML_TYPE_Q5_K);

        assert_eq!(plan(Ftype::Q4KS, "blk.12.ffn_down.weight"), GGML_TYPE_Q4_K);
        assert_eq!(plan(Ftype::Q4KM, "blk.0.ffn_down.weight"), GGML_TYPE_Q6_K);
        assert_eq!(plan(Ftype::Q5KM, "blk.0.attn_v.weight"), GGML_TYPE_Q6_K);
        assert_eq!(plan(Ftype::Q5KS, "blk.0.attn_v.weight"), GGML_TYPE_Q5_K);

        // Two bits never lands on the tensors that cannot take it.
        assert_eq!(plan(Ftype::Q2K, "blk.12.ffn_down.weight"), GGML_TYPE_Q3_K);
        assert_eq!(plan(Ftype::Q2K, "blk.12.attn_v.weight"), GGML_TYPE_Q4_K);
        assert_eq!(plan(Ftype::Q2K, "blk.12.attn_q.weight"), GGML_TYPE_Q2_K);

        // The vocabulary projection is six bits in every quantized file,
        // and the plain body type everywhere it is not named.
        for ftype in Ftype::ALL {
            let output = plan(ftype, "output.weight");
            let body = plan(ftype, "blk.12.attn_q.weight");
            if ftype.is_quantized() {
                assert_eq!(output, GGML_TYPE_Q6_K, "{}", ftype.name());
                assert_eq!(body, ftype.base_type(), "{}", ftype.name());
            } else {
                assert_eq!(output, ftype.base_type(), "{}", ftype.name());
            }
        }
    }

    /// Grouped-query attention makes the value projection cheap to carry
    /// high, and the rules only spend there when the ratio earns it.
    #[test]
    fn grouped_query_attention_changes_where_the_bits_go() {
        let grouped = Model { layers: 24, gqa: 4 };
        let plain = Model { layers: 24, gqa: 1 };
        let at =
            |model| plan_tensor(Ftype::Q2K, "blk.12.attn_v.weight", &[2048, 512], model).ggml_type;
        assert_eq!(at(grouped), GGML_TYPE_Q4_K);
        assert_eq!(at(plain), GGML_TYPE_Q3_K);
    }

    #[test]
    fn a_quantized_source_is_refused_by_name() {
        let err = decode(GGML_TYPE_Q4_K, &[0; 144], 256)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Q4_K"), "{err}");
        assert!(err.contains("BF16"), "{err}");
    }

    #[test]
    fn file_types_parse_by_every_spelling_they_are_offered_under() {
        assert_eq!(Ftype::parse("bf16").unwrap(), Ftype::Bf16);
        assert_eq!(Ftype::parse("q4_k_m").unwrap(), Ftype::Q4KM);
        assert_eq!(Ftype::parse("Q3_K_L").unwrap(), Ftype::Q3KL);
        assert_eq!(Ftype::parse("iq4_xs").unwrap(), Ftype::IQ4XS);
        // Every name this tool prints has to parse back.
        for ftype in Ftype::ALL {
            assert_eq!(Ftype::parse(ftype.name()).unwrap(), ftype);
        }
    }

    /// A name the format defines but this tool will not write should say
    /// why, not "unknown".
    #[test]
    fn the_types_that_are_not_written_say_why() {
        let refused = |name: &str| Ftype::parse(name).unwrap_err().to_string();

        let iq2 = refused("iq2_xs");
        assert!(iq2.contains("importance matrix"), "{iq2}");

        let q8 = refused("q8_0");
        assert!(q8.contains("Q6_K"), "{q8}");

        let nonsense = refused("q9_z");
        assert!(nonsense.contains("unknown"), "{nonsense}");
    }
}
