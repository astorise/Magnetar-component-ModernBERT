//! WebAssembly Component producing a single-sequence ModernBERT encoder graph.

wit_bindgen::generate!({
    path: "wit/model-component-graph.wit",
    world: "encoder-component-graph-producer",
});

use magnetar::model_component_graph::graph_builder::{
    Attribute, AttributeValue, GraphPhase, TensorShape, add_node, begin_graph, declare_input,
    finish_graph, weight_edge,
};
use magnetar::model_component_graph::model_config::{self, EncoderSettings};

fn shape(dimensions: Vec<u64>) -> TensorShape {
    TensorShape { dimensions }
}

fn attr(name: &str, value: AttributeValue) -> Attribute {
    Attribute {
        name: name.to_string(),
        value,
    }
}

fn norm(name: &str, input: String, tokens: u64, hidden: u64, encoder: &EncoderSettings) -> String {
    let mut inputs = vec![input, weight_edge(&format!("{name}.weight"))];
    if encoder.norm_bias {
        inputs.push(weight_edge(&format!("{name}.bias")));
    }
    add_node(
        name,
        "affine-layernorm",
        "Normalization",
        &[attr(
            "epsilon",
            AttributeValue::FloatValue(encoder.norm_epsilon),
        )],
        &inputs,
        &shape(vec![tokens, hidden]),
        None,
    )
}

fn linear(name: &str, input: String, tokens: u64, output: u64, with_bias: bool) -> String {
    let weights = weight_edge(&format!("{name}.weight"));
    let projected = add_node(
        &format!("{name}.matmul"),
        "matmul",
        "LinearAlgebra",
        &[],
        &[input, weights],
        &shape(vec![tokens, output]),
        None,
    );
    if !with_bias {
        return projected;
    }
    let bias = weight_edge(&format!("{name}.bias"));
    add_node(
        &format!("{name}.bias_add"),
        "add",
        "Tensor",
        &[],
        &[projected, bias],
        &shape(vec![tokens, output]),
        None,
    )
}

fn slice(name: &str, input: String, tokens: u64, start: u64, width: u64) -> String {
    add_node(
        name,
        "slice-last",
        "Tensor",
        &[
            attr("start", AttributeValue::Integer(start as i64)),
            attr("width", AttributeValue::Integer(width as i64)),
        ],
        &[input],
        &shape(vec![tokens, width]),
        None,
    )
}

fn rope(
    name: &str,
    input: String,
    tokens: u64,
    hidden: u64,
    heads: u32,
    dimension: u64,
    theta: f64,
) -> String {
    add_node(
        name,
        "rope",
        "PositionEncoding",
        &[
            attr("base", AttributeValue::FloatValue(theta)),
            attr("dimension", AttributeValue::Integer(dimension as i64)),
            attr("head_count", AttributeValue::Integer(heads as i64)),
            attr("position_offset", AttributeValue::Integer(0)),
            attr(
                "rotation_pairing",
                AttributeValue::Text("half_split".into()),
            ),
        ],
        &[input],
        &shape(vec![tokens, hidden]),
        None,
    )
}

fn binary(
    name: &str,
    operator: &str,
    left: String,
    right: String,
    tokens: u64,
    width: u64,
) -> String {
    add_node(
        name,
        operator,
        "Tensor",
        &[],
        &[left, right],
        &shape(vec![tokens, width]),
        None,
    )
}

fn build_graph(tokens: u64) -> String {
    let config = model_config::model_config();
    let encoder = model_config::encoder_config().expect("encoder configuration is required");
    let activation = model_config::hidden_activation();
    assert_eq!(
        activation.as_deref(),
        Some("gelu"),
        "ModernBERT requires erf-defined GELU"
    );
    assert!(
        config.rope_scaling_factor.is_none(),
        "scaled RoPE is not supported"
    );
    assert!(
        model_config::rope_scaling().is_none(),
        "scaled RoPE is not supported"
    );
    assert!(
        model_config::rope_layers().is_none(),
        "per-layer no-RoPE is not supported"
    );
    assert!(
        model_config::layer_layout().is_none(),
        "decoder layer layout is not supported"
    );
    assert!(
        model_config::mixture_of_experts().is_none(),
        "experts are not supported"
    );
    assert!(
        model_config::mixture_of_experts_layers().is_none(),
        "mixed experts are not supported"
    );
    assert!(
        model_config::expert_routing().is_none(),
        "expert routing is not supported"
    );
    assert!(
        model_config::expert_normalization().is_none(),
        "expert normalization is not supported"
    );
    assert!(
        model_config::multi_head_latent_attention().is_none(),
        "latent attention is not supported"
    );
    assert!(
        model_config::latent_norm_epsilon().is_none(),
        "latent normalization is not supported"
    );
    let options = model_config::options();
    assert!(
        options.final_logit_softcapping.is_none() && options.per_layer_input_size.is_none(),
        "decoder options are not supported"
    );

    let hidden = config.hidden_size;
    let intermediate = config.intermediate_size;
    let heads = config.num_attention_heads;
    let dimension = config.head_dimension;
    assert!(tokens > 0 && tokens <= encoder.max_positions && encoder.max_positions <= (1 << 24));
    assert!(hidden > 0 && hidden <= 16_384 && intermediate > 0 && intermediate <= 131_072);
    assert!(
        heads > 0 && config.num_key_value_heads == heads && dimension > 0 && dimension % 2 == 0
    );
    assert_eq!(dimension.checked_mul(heads as u64), Some(hidden));
    assert!(config.vocabulary_size > 0 && config.vocabulary_size <= (1 << 24));
    assert!(
        encoder.norm_epsilon.is_finite()
            && encoder.norm_epsilon > 0.0
            && (encoder.norm_epsilon as f32).is_finite()
            && (encoder.norm_epsilon as f32) > 0.0
    );
    assert_eq!(encoder.attention_bias, config.attention_bias);
    assert_eq!(encoder.layers.len(), config.num_hidden_layers as usize);
    assert!(!encoder.layers.is_empty() && encoder.layers.len() <= 4096);
    assert!(
        encoder
            .layers
            .iter()
            .enumerate()
            .all(
                |(index, layer)| layer.attention_normalization == (index != 0)
                    && layer.local_radius != Some(0)
                    && layer.rope_theta.is_finite()
                    && layer.rope_theta > 0.0
                    && (layer.rope_theta as f32).is_finite()
                    && (layer.rope_theta as f32) > 0.0
            )
    );

    begin_graph(GraphPhase::Encode, tokens, 0);
    let ids = declare_input("token_ids", &shape(vec![tokens]));
    let mask = declare_input("attention_mask", &shape(vec![tokens]));
    let table = weight_edge("embeddings.tok_embeddings.weight");
    let embedded = add_node(
        "embeddings.lookup",
        "embedding",
        "Tensor",
        &[],
        &[ids, table],
        &shape(vec![tokens, hidden]),
        None,
    );
    let mut state = norm("embeddings.norm", embedded, tokens, hidden, &encoder);
    for (index, layer) in encoder.layers.iter().enumerate() {
        let prefix = format!("layers.{index}");
        let normalized = if layer.attention_normalization {
            norm(
                &format!("{prefix}.attn_norm"),
                state.clone(),
                tokens,
                hidden,
                &encoder,
            )
        } else {
            state.clone()
        };
        let qkv = linear(
            &format!("{prefix}.attn.Wqkv"),
            normalized,
            tokens,
            3 * hidden,
            encoder.attention_bias,
        );
        let q = slice(&format!("{prefix}.q"), qkv.clone(), tokens, 0, hidden);
        let k = slice(&format!("{prefix}.k"), qkv.clone(), tokens, hidden, hidden);
        let v = slice(&format!("{prefix}.v"), qkv, tokens, 2 * hidden, hidden);
        let q = rope(
            &format!("{prefix}.rope_q"),
            q,
            tokens,
            hidden,
            heads,
            dimension,
            layer.rope_theta,
        );
        let k = rope(
            &format!("{prefix}.rope_k"),
            k,
            tokens,
            hidden,
            heads,
            dimension,
            layer.rope_theta,
        );
        let mut attrs = vec![
            attr("head_count", AttributeValue::Integer(heads as i64)),
            attr("head_dimension", AttributeValue::Integer(dimension as i64)),
        ];
        if let Some(radius) = layer.local_radius {
            attrs.push(attr("local_radius", AttributeValue::Integer(radius as i64)));
        }
        let attention = add_node(
            &format!("{prefix}.attention"),
            "encoder-attention",
            "Attention",
            &attrs,
            &[q, k, v, mask.clone()],
            &shape(vec![tokens, hidden]),
            None,
        );
        let attention = linear(
            &format!("{prefix}.attn.Wo"),
            attention,
            tokens,
            hidden,
            encoder.attention_bias,
        );
        state = binary(
            &format!("{prefix}.attention_residual"),
            "residual-add",
            state,
            attention,
            tokens,
            hidden,
        );
        let normalized = norm(
            &format!("{prefix}.mlp_norm"),
            state.clone(),
            tokens,
            hidden,
            &encoder,
        );
        let projected = linear(
            &format!("{prefix}.mlp.Wi"),
            normalized,
            tokens,
            2 * intermediate,
            encoder.mlp_bias,
        );
        let activated = slice(
            &format!("{prefix}.mlp.input"),
            projected.clone(),
            tokens,
            0,
            intermediate,
        );
        let gate = slice(
            &format!("{prefix}.mlp.gate"),
            projected,
            tokens,
            intermediate,
            intermediate,
        );
        let activated = add_node(
            &format!("{prefix}.mlp.gelu"),
            "gelu-exact",
            "Activation",
            &[],
            &[activated],
            &shape(vec![tokens, intermediate]),
            None,
        );
        let gated = binary(
            &format!("{prefix}.mlp.mul"),
            "mul",
            activated,
            gate,
            tokens,
            intermediate,
        );
        let output = linear(
            &format!("{prefix}.mlp.Wo"),
            gated,
            tokens,
            hidden,
            encoder.mlp_bias,
        );
        state = binary(
            &format!("{prefix}.mlp.residual"),
            "residual-add",
            state,
            output,
            tokens,
            hidden,
        );
    }
    let output = norm("final_norm", state, tokens, hidden, &encoder);
    finish_graph(&output)
}

struct Component;

impl Guest for Component {
    fn build_encoder_graph(sequence_length: u64) -> String {
        build_graph(sequence_length)
    }
}

export!(Component);
