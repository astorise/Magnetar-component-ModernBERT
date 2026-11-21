//! ModernBERT encoder layer planning and CPU reference for Magnetar.
//!
//! This library preserves checkpoint attention layout and can evaluate a
//! supported f32 encoder with caller-supplied weights. It does not yet load
//! checkpoint weights or produce Magnetar graphs.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

pub mod encoder_reference;

const MAX_LAYERS: usize = 4096;

/// Both attention variants are bidirectional, not causal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionKind {
    Global,
    Local,
}

/// A layer's checkpoint-derived attention and pre-attention normalization.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerPlan {
    pub attention: AttentionKind,
    pub rope_theta: f64,
    /// ModernBERT uses identity rather than attention LayerNorm on layer 0.
    pub has_attention_norm: bool,
}

/// The validated encoder dimensions and layer order from a checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderPlan {
    pub hidden_size: usize,
    pub attention_heads: usize,
    /// The upstream `local_attention` parameter, not a causal KV window.
    pub local_attention: usize,
    pub layers: Vec<LayerPlan>,
}

#[derive(Deserialize)]
struct RawConfig {
    model_type: String,
    hidden_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    layer_types: Option<Vec<String>>,
    global_attn_every_n_layers: Option<usize>,
    local_attention: Option<usize>,
    global_rope_theta: Option<f64>,
    local_rope_theta: Option<f64>,
    rope_scaling: Option<serde_json::Value>,
    rope_parameters: Option<BTreeMap<String, RopeParameters>>,
}

#[derive(Deserialize)]
struct RopeParameters {
    rope_type: Option<String>,
    rope_theta: Option<f64>,
}

impl EncoderPlan {
    /// Parse only the encoder attention layout from a Hugging Face config.
    /// Unknown attention and scaled RoPE variants are rejected. A successful
    /// plan does not mean a checkpoint is executable by Magnetar.
    pub fn from_huggingface_config(bytes: &[u8]) -> Result<Self, ConfigError> {
        let config: RawConfig = serde_json::from_slice(bytes)
            .map_err(|error| ConfigError(format!("invalid ModernBERT config: {error}")))?;
        if config.model_type != "modernbert" {
            return Err(ConfigError(format!(
                "expected model_type modernbert, got {:?}",
                config.model_type
            )));
        }
        if !(1..=MAX_LAYERS).contains(&config.num_hidden_layers) {
            return Err(ConfigError(format!(
                "num_hidden_layers must be between 1 and {MAX_LAYERS}"
            )));
        }
        if config.hidden_size == 0
            || config.num_attention_heads == 0
            || !config
                .hidden_size
                .is_multiple_of(config.num_attention_heads)
            || !(config.hidden_size / config.num_attention_heads).is_multiple_of(2)
        {
            return Err(ConfigError(
                "attention heads require an even, integral head size".into(),
            ));
        }
        let local_attention = config.local_attention.unwrap_or(128);
        if local_attention < 2 {
            return Err(ConfigError("local_attention must be at least 2".into()));
        }
        if config.rope_scaling.is_some() {
            return Err(ConfigError(
                "scaled RoPE is not supported by this planner".into(),
            ));
        }
        let parameters = config.rope_parameters.unwrap_or_default();
        let theta = |name: &str, legacy: Option<f64>, default: f64| {
            let specific = parameters.get(name);
            if specific
                .and_then(|p| p.rope_type.as_deref())
                .unwrap_or("default")
                != "default"
            {
                return Err(ConfigError(format!("unsupported RoPE type for {name}")));
            }
            let value = specific
                .and_then(|p| p.rope_theta)
                .or(legacy)
                .unwrap_or(default);
            if !value.is_finite() || value <= 0.0 {
                return Err(ConfigError(format!("invalid RoPE theta for {name}")));
            }
            Ok(value)
        };
        let global_theta = theta("full_attention", config.global_rope_theta, 160_000.0)?;
        let local_theta = theta("sliding_attention", config.local_rope_theta, 10_000.0)?;

        let kinds = if let Some(types) = config.layer_types {
            if types.len() != config.num_hidden_layers {
                return Err(ConfigError(
                    "layer_types length must equal num_hidden_layers".into(),
                ));
            }
            types
                .iter()
                .map(|kind| match kind.as_str() {
                    "full_attention" => Ok(AttentionKind::Global),
                    "sliding_attention" => Ok(AttentionKind::Local),
                    _ => Err(ConfigError(format!("unsupported layer type {kind:?}"))),
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            let interval = config.global_attn_every_n_layers.unwrap_or(3);
            if interval == 0 {
                return Err(ConfigError(
                    "global attention interval must be nonzero".into(),
                ));
            }
            (0..config.num_hidden_layers)
                .map(|index| {
                    if index % interval == 0 {
                        AttentionKind::Global
                    } else {
                        AttentionKind::Local
                    }
                })
                .collect()
        };
        let layers = kinds
            .into_iter()
            .enumerate()
            .map(|(index, attention)| LayerPlan {
                attention,
                rope_theta: match attention {
                    AttentionKind::Global => global_theta,
                    AttentionKind::Local => local_theta,
                },
                has_attention_norm: index != 0,
            })
            .collect();
        Ok(Self {
            hidden_size: config.hidden_size,
            attention_heads: config.num_attention_heads,
            local_attention,
            layers,
        })
    }
}

/// A malformed or unsupported ModernBERT configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(String);

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ConfigError {}
