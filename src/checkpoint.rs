//! Checkpoint-derived ModernBERT geometry and weight mapping.
//!
//! The caller authenticates and decodes the checkpoint. This module validates
//! its encoder tensors, constructs the CPU reference, and describes the
//! transposition required by the Component's graph weight edges. It neither
//! reads a checkpoint file nor loads a model into the Runtime.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

use crate::{
    ConfigError, EncoderPlan,
    encoder_reference::{Encoder, EncoderLayer, LayerNorm, Linear},
};

const MAX_ELEMENTS: usize = 67_108_864;

/// A verified, decoded f32 tensor in checkpoint order.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedTensor {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

/// Storage types that the caller can decode to the f32 execution profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageDType {
    F16,
    BF16,
    F32,
    /// A format that this profile cannot decode faithfully to f32.
    Unsupported,
}

/// Metadata from a previously validated checkpoint index or header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorDescriptor {
    pub shape: Vec<usize>,
    pub dtype: StorageDType,
}

/// Values required to bind the existing encoder graph configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphSettings {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocabulary_size: usize,
    pub attention_heads: usize,
    pub max_positions: usize,
    pub norm_epsilon: f64,
    pub norm_bias: bool,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    pub layers: Vec<GraphLayerSettings>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphLayerSettings {
    /// Inclusive symmetric window radius, or `None` for global attention.
    pub local_radius: Option<usize>,
    pub rope_theta: f64,
    pub attention_normalization: bool,
}

/// Mapping from a checkpoint tensor to the encoder graph's weight edge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightSpec {
    pub checkpoint_name: String,
    pub graph_name: String,
    pub checkpoint_shape: Vec<usize>,
    pub graph_shape: Vec<usize>,
    pub transpose: bool,
}

impl WeightSpec {
    /// Convert a decoded tensor to graph order, checking shape and values.
    pub fn graph_values(&self, tensor: &DecodedTensor) -> Result<Vec<f32>, ConfigError> {
        validate_tensor(self, tensor)?;
        if !self.transpose {
            if self.graph_shape != self.checkpoint_shape {
                return Err(ConfigError("invalid identity weight mapping".into()));
            }
            return Ok(tensor.values.clone());
        }
        let [rows, columns] = self.checkpoint_shape.as_slice() else {
            return Err(ConfigError("invalid matrix weight mapping".into()));
        };
        if self.graph_shape != [*columns, *rows] {
            return Err(ConfigError("invalid transposed weight mapping".into()));
        }
        let mut values = vec![0.0; tensor.values.len()];
        for row in 0..*rows {
            for column in 0..*columns {
                values[column * rows + row] = tensor.values[row * columns + column];
            }
        }
        Ok(values)
    }
}

/// The supported full encoder profile extracted from a checkpoint config.
#[derive(Clone, Debug)]
pub struct CheckpointProfile {
    layout: EncoderPlan,
    vocabulary: usize,
    intermediate: usize,
    max_positions: usize,
    epsilon: f64,
    norm_bias: bool,
    attention_bias: bool,
    mlp_bias: bool,
    prefix: &'static str,
    weights: Vec<WeightSpec>,
}

#[derive(Deserialize)]
struct RawConfig {
    vocab_size: usize,
    intermediate_size: usize,
    max_position_embeddings: usize,
    norm_eps: f64,
    hidden_activation: String,
    norm_bias: bool,
    attention_bias: bool,
    mlp_bias: bool,
}

/// Location of the encoder tensors in the source checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointNamespace {
    /// The checkpoint contains only the ModernBERT model (plus optional heads).
    Root,
    /// Laya's decision checkpoint nests the encoder under `encoder.`.
    Encoder,
}

impl CheckpointProfile {
    /// Parse a default-RoPE, erf-GELU encoder config and its checkpoint layout.
    ///
    /// Every setting needed by the graph is required rather than guessed.
    /// The returned profile describes f32 computation; storage decoding and
    /// provenance verification remain the caller's responsibility.
    pub fn from_huggingface_config(
        bytes: &[u8],
        namespace: CheckpointNamespace,
    ) -> Result<Self, ConfigError> {
        let layout = EncoderPlan::from_huggingface_config(bytes)?;
        let raw: RawConfig = serde_json::from_slice(bytes)
            .map_err(|error| ConfigError(format!("invalid encoder checkpoint config: {error}")))?;
        let epsilon = raw.norm_eps as f32;
        if raw.hidden_activation != "gelu"
            || raw.vocab_size == 0
            || raw.vocab_size > (1 << 24)
            || layout.hidden_size > 4096
            || raw.intermediate_size == 0
            || raw.intermediate_size > 131_072
            || raw.max_position_embeddings == 0
            || raw.max_position_embeddings > (1 << 24)
            || layout.local_attention / 2 > u32::MAX as usize
            || !raw.norm_eps.is_finite()
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || layout.layers.iter().any(|layer| {
                let theta = layer.rope_theta as f32;
                !theta.is_finite() || theta <= 0.0
            })
        {
            return Err(ConfigError(
                "unsupported encoder checkpoint geometry or numerics".into(),
            ));
        }
        let prefix = match namespace {
            CheckpointNamespace::Root => "",
            CheckpointNamespace::Encoder => "encoder.",
        };
        let mut result = Self {
            layout,
            vocabulary: raw.vocab_size,
            intermediate: raw.intermediate_size,
            max_positions: raw.max_position_embeddings,
            epsilon: raw.norm_eps,
            norm_bias: raw.norm_bias,
            attention_bias: raw.attention_bias,
            mlp_bias: raw.mlp_bias,
            prefix,
            weights: Vec::new(),
        };
        result.weights = result.make_weights()?;
        Ok(result)
    }

    pub fn layout(&self) -> &EncoderPlan {
        &self.layout
    }

    /// Checkpoint-derived values for the existing encoder graph capability.
    pub fn graph_settings(&self) -> GraphSettings {
        GraphSettings {
            hidden_size: self.layout.hidden_size,
            intermediate_size: self.intermediate,
            vocabulary_size: self.vocabulary,
            attention_heads: self.layout.attention_heads,
            max_positions: self.max_positions,
            norm_epsilon: self.epsilon,
            norm_bias: self.norm_bias,
            attention_bias: self.attention_bias,
            mlp_bias: self.mlp_bias,
            layers: self
                .layout
                .layers
                .iter()
                .map(|layer| GraphLayerSettings {
                    local_radius: (layer.attention == crate::AttentionKind::Local)
                        .then_some(self.layout.local_attention / 2),
                    rope_theta: layer.rope_theta,
                    attention_normalization: layer.has_attention_norm,
                })
                .collect(),
        }
    }

    /// Required tensors in checkpoint order, with the graph order alongside.
    pub fn weights(&self) -> &[WeightSpec] {
        &self.weights
    }

    /// Check names, shapes and storage types before decoding tensor payloads.
    /// Payload integrity and the authenticity of the index are caller-owned.
    pub fn validate_inventory(
        &self,
        inventory: &BTreeMap<String, TensorDescriptor>,
    ) -> Result<(), ConfigError> {
        for spec in &self.weights {
            let descriptor = inventory.get(&spec.checkpoint_name).ok_or_else(|| {
                ConfigError(format!("missing encoder tensor {}", spec.checkpoint_name))
            })?;
            if descriptor.shape != spec.checkpoint_shape
                || descriptor.dtype == StorageDType::Unsupported
            {
                return Err(ConfigError(format!(
                    "invalid encoder tensor metadata {}",
                    spec.checkpoint_name
                )));
            }
        }
        self.validate_names(inventory.keys().map(String::as_str))
    }

    /// Validate decoded tensors without accepting missing, malformed, nonfinite
    /// or unexpected encoder weights. Non-encoder decision/model heads may be
    /// present in the same checkpoint and are ignored.
    pub fn validate_tensors(
        &self,
        tensors: &BTreeMap<String, DecodedTensor>,
    ) -> Result<(), ConfigError> {
        for spec in &self.weights {
            let tensor = tensors.get(&spec.checkpoint_name).ok_or_else(|| {
                ConfigError(format!("missing encoder tensor {}", spec.checkpoint_name))
            })?;
            validate_tensor(spec, tensor)?;
        }
        self.validate_names(tensors.keys().map(String::as_str))
    }

    fn validate_names<'a>(&self, names: impl Iterator<Item = &'a str>) -> Result<(), ConfigError> {
        let expected: BTreeSet<&str> = self
            .weights
            .iter()
            .map(|spec| spec.checkpoint_name.as_str())
            .collect();
        for name in names {
            let encoder_name = name.strip_prefix(self.prefix).is_some_and(|name| {
                name.starts_with("embeddings.")
                    || name.starts_with("layers.")
                    || name.starts_with("final_norm.")
                    || (!self.prefix.is_empty() && !name.is_empty())
            });
            if encoder_name && !expected.contains(name) {
                return Err(ConfigError(format!("unexpected encoder tensor {name}")));
            }
        }
        Ok(())
    }

    /// Construct the single-sequence reference using decoded checkpoint-order
    /// weights. It does not register a Component or execute a Runtime graph.
    pub fn into_reference(
        &self,
        mut tensors: BTreeMap<String, DecodedTensor>,
    ) -> Result<Encoder, ConfigError> {
        self.validate_tensors(&tensors)?;
        let mut take = |name: &str| -> Result<Vec<f32>, ConfigError> {
            tensors
                .remove(&format!("{}{name}", self.prefix))
                .map(|tensor| tensor.values)
                .ok_or_else(|| ConfigError(format!("missing encoder tensor {name}")))
        };
        fn norm(
            take: &mut impl FnMut(&str) -> Result<Vec<f32>, ConfigError>,
            name: &str,
            bias: bool,
        ) -> Result<LayerNorm, ConfigError> {
            Ok(LayerNorm {
                weight: take(&format!("{name}.weight"))?,
                bias: if bias {
                    Some(take(&format!("{name}.bias"))?)
                } else {
                    None
                },
            })
        }
        fn linear(
            take: &mut impl FnMut(&str) -> Result<Vec<f32>, ConfigError>,
            name: &str,
            bias: bool,
        ) -> Result<Linear, ConfigError> {
            Ok(Linear {
                weight: take(&format!("{name}.weight"))?,
                bias: if bias {
                    Some(take(&format!("{name}.bias"))?)
                } else {
                    None
                },
            })
        }

        let token_embedding = take("embeddings.tok_embeddings.weight")?;
        let embedding_norm = norm(&mut take, "embeddings.norm", self.norm_bias)?;
        let mut layers = Vec::with_capacity(self.layout.layers.len());
        for (index, plan) in self.layout.layers.iter().enumerate() {
            let prefix = format!("layers.{index}");
            layers.push(EncoderLayer {
                attention: plan.attention,
                rope_theta: plan.rope_theta as f32,
                attention_norm: if plan.has_attention_norm {
                    Some(norm(
                        &mut take,
                        &format!("{prefix}.attn_norm"),
                        self.norm_bias,
                    )?)
                } else {
                    None
                },
                qkv: linear(
                    &mut take,
                    &format!("{prefix}.attn.Wqkv"),
                    self.attention_bias,
                )?,
                attention_output: linear(
                    &mut take,
                    &format!("{prefix}.attn.Wo"),
                    self.attention_bias,
                )?,
                mlp_norm: norm(&mut take, &format!("{prefix}.mlp_norm"), self.norm_bias)?,
                mlp_in: linear(&mut take, &format!("{prefix}.mlp.Wi"), self.mlp_bias)?,
                mlp_out: linear(&mut take, &format!("{prefix}.mlp.Wo"), self.mlp_bias)?,
            });
        }
        let final_norm = norm(&mut take, "final_norm", self.norm_bias)?;
        Ok(Encoder {
            vocab_size: self.vocabulary,
            hidden: self.layout.hidden_size,
            intermediate: self.intermediate,
            heads: self.layout.attention_heads,
            max_positions: self.max_positions,
            local_radius: self.layout.local_attention / 2,
            epsilon: self.epsilon as f32,
            token_embedding,
            embedding_norm,
            layers,
            final_norm,
        })
    }

    fn make_weights(&self) -> Result<Vec<WeightSpec>, ConfigError> {
        let mut result = Vec::new();
        let mut add =
            |name: String, shape: Vec<usize>, transpose: bool| -> Result<(), ConfigError> {
                element_count(&shape)?;
                let graph_shape = if transpose {
                    vec![shape[1], shape[0]]
                } else {
                    shape.clone()
                };
                result.push(WeightSpec {
                    checkpoint_name: format!("{}{name}", self.prefix),
                    graph_name: name,
                    checkpoint_shape: shape,
                    graph_shape,
                    transpose,
                });
                Ok(())
            };
        let h = self.layout.hidden_size;
        let i = self.intermediate;
        add(
            "embeddings.tok_embeddings.weight".into(),
            vec![self.vocabulary, h],
            false,
        )?;
        let norm = |add: &mut dyn FnMut(String, Vec<usize>, bool) -> Result<(), ConfigError>,
                    name: &str| {
            add(format!("{name}.weight"), vec![h], false)?;
            if self.norm_bias {
                add(format!("{name}.bias"), vec![h], false)?;
            }
            Ok(())
        };
        let linear = |add: &mut dyn FnMut(String, Vec<usize>, bool) -> Result<(), ConfigError>,
                      name: &str,
                      input,
                      output,
                      bias| {
            add(format!("{name}.weight"), vec![output, input], true)?;
            if bias {
                add(format!("{name}.bias"), vec![output], false)?;
            }
            Ok(())
        };
        norm(&mut add, "embeddings.norm")?;
        for (index, plan) in self.layout.layers.iter().enumerate() {
            let prefix = format!("layers.{index}");
            if plan.has_attention_norm {
                norm(&mut add, &format!("{prefix}.attn_norm"))?;
            }
            linear(
                &mut add,
                &format!("{prefix}.attn.Wqkv"),
                h,
                3 * h,
                self.attention_bias,
            )?;
            linear(
                &mut add,
                &format!("{prefix}.attn.Wo"),
                h,
                h,
                self.attention_bias,
            )?;
            norm(&mut add, &format!("{prefix}.mlp_norm"))?;
            linear(
                &mut add,
                &format!("{prefix}.mlp.Wi"),
                h,
                2 * i,
                self.mlp_bias,
            )?;
            linear(&mut add, &format!("{prefix}.mlp.Wo"), i, h, self.mlp_bias)?;
        }
        norm(&mut add, "final_norm")?;
        Ok(result)
    }
}

fn element_count(shape: &[usize]) -> Result<usize, ConfigError> {
    if shape.is_empty() {
        return Err(ConfigError("empty encoder tensor shape".into()));
    }
    shape.iter().try_fold(1usize, |count, &dimension| {
        count
            .checked_mul(dimension)
            .filter(|&value| dimension > 0 && value <= MAX_ELEMENTS)
            .ok_or_else(|| ConfigError("encoder tensor exceeds reference limits".into()))
    })
}

fn validate_tensor(spec: &WeightSpec, tensor: &DecodedTensor) -> Result<(), ConfigError> {
    if tensor.shape != spec.checkpoint_shape
        || element_count(&tensor.shape)? != tensor.values.len()
        || tensor.values.iter().any(|value| !value.is_finite())
    {
        return Err(ConfigError(format!(
            "invalid encoder tensor {}",
            spec.checkpoint_name
        )));
    }
    Ok(())
}
