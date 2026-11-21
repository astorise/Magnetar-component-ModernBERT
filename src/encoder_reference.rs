//! Single-sequence f32 ModernBERT encoder reference for supplied weights.
//!
//! This is an inference oracle, not a Magnetar Component graph or checkpoint
//! loader. It implements the pinned default-RoPE GELU encoder profile.

use crate::AttentionKind;
use std::{error::Error, fmt};

const MAX_ELEMENTS: usize = 67_108_864;

/// PyTorch output-major linear weight and optional bias.
#[derive(Clone, Debug)]
pub struct Linear {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

/// Learned LayerNorm with a checkpoint-selected optional bias.
#[derive(Clone, Debug)]
pub struct LayerNorm {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

/// One checkpoint layer, in its original order.
#[derive(Clone, Debug)]
pub struct EncoderLayer {
    pub attention: AttentionKind,
    pub rope_theta: f32,
    /// The first layer has no attention normalization.
    pub attention_norm: Option<LayerNorm>,
    pub qkv: Linear,
    pub attention_output: Linear,
    pub mlp_norm: LayerNorm,
    /// `[2 * intermediate, hidden]`, activation branch first.
    pub mlp_in: Linear,
    pub mlp_out: Linear,
}

/// Weights and geometry of one default-RoPE ModernBERT encoder.
#[derive(Clone, Debug)]
pub struct Encoder {
    pub vocab_size: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub max_positions: usize,
    /// Inclusive radius for layers with local attention.
    pub local_radius: usize,
    pub epsilon: f32,
    /// `[vocab_size, hidden]` in token-ID order.
    pub token_embedding: Vec<f32>,
    pub embedding_norm: LayerNorm,
    pub layers: Vec<EncoderLayer>,
    pub final_norm: LayerNorm,
}

/// Invalid weights, inputs, position IDs, or numeric values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncoderError(pub String);

impl fmt::Display for EncoderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}
impl Error for EncoderError {}

fn product(a: usize, b: usize) -> Result<usize, EncoderError> {
    a.checked_mul(b)
        .filter(|&value| value <= MAX_ELEMENTS)
        .ok_or_else(|| EncoderError("encoder dimensions exceed reference limits".into()))
}

fn finite(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

impl Linear {
    fn validate(&self, input: usize, output: usize) -> Result<(), EncoderError> {
        if self.weight.len() != product(input, output)?
            || !finite(&self.weight)
            || self
                .bias
                .as_ref()
                .is_some_and(|bias| bias.len() != output || !finite(bias))
        {
            return Err(EncoderError("invalid encoder linear parameters".into()));
        }
        Ok(())
    }

    fn apply(&self, input: &[f32]) -> Vec<f32> {
        self.weight
            .chunks_exact(input.len())
            .enumerate()
            .map(|(row, weights)| {
                weights.iter().zip(input).fold(
                    self.bias.as_ref().map_or(0.0, |bias| bias[row]),
                    |sum, (&weight, &value)| sum + weight * value,
                )
            })
            .collect()
    }
}

impl LayerNorm {
    fn validate(&self, hidden: usize) -> Result<(), EncoderError> {
        if self.weight.len() != hidden
            || !finite(&self.weight)
            || self
                .bias
                .as_ref()
                .is_some_and(|bias| bias.len() != hidden || !finite(bias))
        {
            return Err(EncoderError(
                "invalid encoder normalization parameters".into(),
            ));
        }
        Ok(())
    }

    fn apply(&self, input: &[f32], epsilon: f32) -> Result<Vec<f32>, EncoderError> {
        let mean = input.iter().sum::<f32>() / input.len() as f32;
        let variance = input
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / input.len() as f32;
        let adjusted = variance + epsilon;
        if !mean.is_finite() || !adjusted.is_finite() {
            return Err(EncoderError("non-finite encoder normalization".into()));
        }
        let inverse = adjusted.sqrt().recip();
        let output: Vec<_> = input
            .iter()
            .zip(&self.weight)
            .enumerate()
            .map(|(index, (&value, &weight))| {
                (value - mean) * inverse * weight
                    + self.bias.as_ref().map_or(0.0, |bias| bias[index])
            })
            .collect();
        if !finite(&output) {
            return Err(EncoderError("non-finite encoder normalized output".into()));
        }
        Ok(output)
    }
}

fn gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + libm::erff(value * std::f32::consts::FRAC_1_SQRT_2))
}

impl Encoder {
    fn validate(&self) -> Result<usize, EncoderError> {
        let d = self.hidden;
        if self.vocab_size == 0
            || d == 0
            || d > 4096
            || self.intermediate == 0
            || self.heads == 0
            || !d.is_multiple_of(self.heads)
            || !(d / self.heads).is_multiple_of(2)
            || self.max_positions == 0
            || self.max_positions > (1 << 24)
            || self.local_radius == 0
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || self.layers.is_empty()
            || self.layers.len() > 4096
            || self.token_embedding.len() != product(self.vocab_size, d)?
            || !finite(&self.token_embedding)
        {
            return Err(EncoderError("invalid encoder geometry or embedding".into()));
        }
        self.embedding_norm.validate(d)?;
        self.final_norm.validate(d)?;
        for (index, layer) in self.layers.iter().enumerate() {
            if !layer.rope_theta.is_finite()
                || layer.rope_theta <= 0.0
                || layer.attention_norm.is_some() != (index != 0)
            {
                return Err(EncoderError(
                    "invalid encoder layer normalization or rotary base".into(),
                ));
            }
            if let Some(norm) = &layer.attention_norm {
                norm.validate(d)?;
            }
            layer.qkv.validate(d, product(3, d)?)?;
            layer.attention_output.validate(d, d)?;
            layer.mlp_norm.validate(d)?;
            layer.mlp_in.validate(d, product(2, self.intermediate)?)?;
            layer.mlp_out.validate(self.intermediate, d)?;
        }
        Ok(d / self.heads)
    }

    /// Encode one sequence, preserving padded query rows as upstream does.
    /// The key mask is applied to both global and local attention; local
    /// windows use physical row distance. Missing positions mean `0..tokens`.
    pub fn forward(
        &self,
        token_ids: &[u32],
        key_mask: &[bool],
        positions: Option<&[u32]>,
    ) -> Result<Vec<f32>, EncoderError> {
        let width = self.validate()?;
        let tokens = token_ids.len();
        let d = self.hidden;
        if tokens == 0
            || tokens > 4096
            || key_mask.len() != tokens
            || !key_mask.contains(&true)
            || token_ids.iter().any(|&id| id as usize >= self.vocab_size)
            || positions.is_some_and(|values| {
                values.len() != tokens
                    || values
                        .iter()
                        .any(|&value| value as usize >= self.max_positions)
            })
            || positions.is_none() && tokens > self.max_positions
        {
            return Err(EncoderError(
                "invalid encoder token, mask or position".into(),
            ));
        }
        product(tokens, product(3, d)?)?;
        product(tokens, product(2, self.intermediate)?)?;
        let mut states = Vec::with_capacity(tokens * d);
        for &id in token_ids {
            let offset = id as usize * d;
            states.extend(
                self.embedding_norm
                    .apply(&self.token_embedding[offset..offset + d], self.epsilon)?,
            );
        }
        for layer in &self.layers {
            let mut qkv = Vec::with_capacity(tokens * 3 * d);
            for row in states.chunks_exact(d) {
                let normalized = match &layer.attention_norm {
                    Some(norm) => norm.apply(row, self.epsilon)?,
                    None => row.to_vec(),
                };
                qkv.extend(layer.qkv.apply(&normalized));
            }
            if !finite(&qkv) {
                return Err(EncoderError("non-finite encoder QKV projection".into()));
            }
            for row in 0..tokens {
                let position = positions.map_or(row as f32, |values| values[row] as f32);
                for head in 0..self.heads {
                    for index in 0..width / 2 {
                        let angle =
                            position * layer.rope_theta.powf(-((2 * index) as f32) / width as f32);
                        let (sin, cos) = angle.sin_cos();
                        for component in 0..2 {
                            let first = row * 3 * d + component * d + head * width + index;
                            let second = first + width / 2;
                            let a = qkv[first];
                            let b = qkv[second];
                            qkv[first] = a * cos - b * sin;
                            qkv[second] = b * cos + a * sin;
                        }
                    }
                }
            }
            let mut attention = vec![0.0; tokens * d];
            for query in 0..tokens {
                let start = if layer.attention == AttentionKind::Local {
                    query.saturating_sub(self.local_radius)
                } else {
                    0
                };
                let end = if layer.attention == AttentionKind::Local {
                    query
                        .saturating_add(self.local_radius)
                        .saturating_add(1)
                        .min(tokens)
                } else {
                    tokens
                };
                for head in 0..self.heads {
                    let mut scores = Vec::with_capacity(end - start);
                    for (key, &valid) in key_mask.iter().enumerate().take(end).skip(start) {
                        if !valid {
                            continue;
                        }
                        let mut score = 0.0;
                        for index in 0..width {
                            score += qkv[query * 3 * d + head * width + index]
                                * qkv[key * 3 * d + d + head * width + index];
                        }
                        score /= (width as f32).sqrt();
                        if !score.is_finite() {
                            return Err(EncoderError("non-finite encoder attention score".into()));
                        }
                        scores.push((key, score));
                    }
                    if scores.is_empty() {
                        return Err(EncoderError(
                            "encoder query has no valid attention key".into(),
                        ));
                    }
                    let max = scores
                        .iter()
                        .map(|(_, score)| *score)
                        .fold(f32::NEG_INFINITY, f32::max);
                    let denominator: f32 =
                        scores.iter().map(|(_, score)| (*score - max).exp()).sum();
                    for index in 0..width {
                        let mut value = 0.0;
                        for &(key, score) in &scores {
                            value += (score - max).exp() / denominator
                                * qkv[key * 3 * d + 2 * d + head * width + index];
                        }
                        attention[query * d + head * width + index] = value;
                    }
                }
            }
            for (row, attended) in states.chunks_exact_mut(d).zip(attention.chunks_exact(d)) {
                for (value, update) in row.iter_mut().zip(layer.attention_output.apply(attended)) {
                    *value += update;
                }
                let normalized = layer.mlp_norm.apply(row, self.epsilon)?;
                let projected = layer.mlp_in.apply(&normalized);
                if !finite(&projected) {
                    return Err(EncoderError("non-finite encoder MLP projection".into()));
                }
                let intermediate: Vec<_> = (0..self.intermediate)
                    .map(|index| gelu(projected[index]) * projected[self.intermediate + index])
                    .collect();
                if !finite(&intermediate) {
                    return Err(EncoderError("non-finite encoder MLP activation".into()));
                }
                for (value, update) in row.iter_mut().zip(layer.mlp_out.apply(&intermediate)) {
                    *value += update;
                }
            }
            if !finite(&states) {
                return Err(EncoderError("non-finite encoder layer output".into()));
            }
        }
        let mut output = Vec::with_capacity(states.len());
        for row in states.chunks_exact(d) {
            output.extend(self.final_norm.apply(row, self.epsilon)?);
        }
        Ok(output)
    }
}
