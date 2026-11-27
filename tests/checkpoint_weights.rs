use std::collections::BTreeMap;

use magnetar_component_modernbert::checkpoint::{
    CheckpointNamespace, CheckpointProfile, DecodedTensor, StorageDType, TensorDescriptor,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    bias: bool,
    ids: Vec<u32>,
    mask: Vec<u8>,
    positions: Vec<u32>,
    output: Vec<f32>,
}

fn config(bias: bool) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "model_type": "modernbert", "hidden_size": 8, "intermediate_size": 12,
        "vocab_size": 24, "num_attention_heads": 2, "num_hidden_layers": 3,
        "max_position_embeddings": 64, "norm_eps": 1e-5,
        "norm_bias": bias, "attention_bias": bias, "mlp_bias": bias,
        "hidden_activation": "gelu", "local_attention": 4,
        "layer_types": ["full_attention", "sliding_attention", "full_attention"],
        "rope_parameters": {
            "full_attention": {"rope_type": "default", "rope_theta": 160000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}
        }
    }))
    .unwrap()
}

// The independent Transformers fixture seeds every checkpoint-order tensor
// with this deterministic name-dependent sequence.
fn pattern(name: &str, count: usize) -> Vec<f32> {
    let seed = name
        .bytes()
        .enumerate()
        .map(|(i, b)| (i + 1) * b as usize)
        .sum::<usize>()
        % 97;
    (0..count)
        .map(|i| {
            let value = ((i * 17 + seed) % 101) as f32 - 50.0;
            value / 200.0
                + if name.contains("norm") && name.ends_with(".weight") {
                    1.0
                } else {
                    0.0
                }
        })
        .collect()
}

fn tensors(profile: &CheckpointProfile) -> BTreeMap<String, DecodedTensor> {
    profile
        .weights()
        .iter()
        .map(|weight| {
            (
                weight.checkpoint_name.clone(),
                DecodedTensor {
                    shape: weight.checkpoint_shape.clone(),
                    values: pattern(&weight.graph_name, weight.checkpoint_shape.iter().product()),
                },
            )
        })
        .collect()
}

#[test]
fn decoded_checkpoint_runs_against_independent_transformers_encoder() {
    let cases: Vec<Case> =
        serde_json::from_str(include_str!("fixtures/encoder_reference.json")).unwrap();
    for case in cases {
        for namespace in [CheckpointNamespace::Root, CheckpointNamespace::Encoder] {
            let profile =
                CheckpointProfile::from_huggingface_config(&config(case.bias), namespace).unwrap();
            let model = profile.into_reference(tensors(&profile)).unwrap();
            let mask: Vec<_> = case.mask.iter().map(|&value| value != 0).collect();
            let actual = model
                .forward(&case.ids, &mask, Some(&case.positions))
                .unwrap();
            assert_eq!(actual.len(), case.output.len());
            for (index, (&actual, &expected)) in actual.iter().zip(&case.output).enumerate() {
                assert!(
                    (actual - expected).abs() <= 2e-6 + 1e-6 * expected.abs(),
                    "bias {} output[{index}] {actual} != {expected}",
                    case.bias
                );
            }
        }
    }
}

#[test]
fn published_laya_config_maps_encoder_weights_to_graph_shapes() {
    let profile = CheckpointProfile::from_huggingface_config(
        include_bytes!("fixtures/laya-encoder.config.json"),
        CheckpointNamespace::Encoder,
    )
    .unwrap();
    assert_eq!(profile.layout().layers.len(), 28);
    assert_eq!(profile.weights().len(), 28 * 5 + 27 + 3);
    let settings = profile.graph_settings();
    assert_eq!(
        (
            settings.hidden_size,
            settings.intermediate_size,
            settings.vocabulary_size
        ),
        (1024, 2624, 50368)
    );
    assert_eq!(
        (settings.attention_heads, settings.max_positions),
        (16, 8192)
    );
    assert_eq!(settings.layers[0].local_radius, None);
    assert!(!settings.layers[0].attention_normalization);
    assert_eq!(settings.layers[1].local_radius, Some(64));
    assert_eq!(settings.layers[1].rope_theta, 10_000.0);
    assert!(settings.layers[1].attention_normalization);
    assert!(!settings.norm_bias && !settings.attention_bias && !settings.mlp_bias);
    let get = |name: &str| {
        profile
            .weights()
            .iter()
            .find(|spec| spec.graph_name == name)
            .unwrap()
    };
    let embedding = get("embeddings.tok_embeddings.weight");
    assert_eq!(
        embedding.checkpoint_name,
        "encoder.embeddings.tok_embeddings.weight"
    );
    assert_eq!(embedding.graph_shape, [50368, 1024]);
    assert!(!embedding.transpose);
    let qkv = get("layers.0.attn.Wqkv.weight");
    assert_eq!(qkv.checkpoint_shape, [3072, 1024]);
    assert_eq!(qkv.graph_shape, [1024, 3072]);
    assert!(qkv.transpose);
    let mlp = get("layers.27.mlp.Wi.weight");
    assert_eq!(mlp.checkpoint_shape, [5248, 1024]);
    assert_eq!(mlp.graph_shape, [1024, 5248]);
    assert!(
        !profile
            .weights()
            .iter()
            .any(|spec| spec.graph_name == "layers.0.attn_norm.weight")
    );
    assert!(
        profile
            .weights()
            .iter()
            .any(|spec| spec.graph_name == "layers.27.attn_norm.weight")
    );
}

#[test]
fn inventory_is_checked_before_weight_payloads_are_decoded() {
    let profile =
        CheckpointProfile::from_huggingface_config(&config(false), CheckpointNamespace::Encoder)
            .unwrap();
    let mut inventory: BTreeMap<_, _> = profile
        .weights()
        .iter()
        .map(|weight| {
            (
                weight.checkpoint_name.clone(),
                TensorDescriptor {
                    shape: weight.checkpoint_shape.clone(),
                    dtype: StorageDType::F16,
                },
            )
        })
        .collect();
    assert!(profile.validate_inventory(&inventory).is_ok());
    let name = "encoder.layers.0.attn.Wqkv.weight";
    inventory.get_mut(name).unwrap().dtype = StorageDType::BF16;
    assert!(profile.validate_inventory(&inventory).is_ok());
    inventory.get_mut(name).unwrap().dtype = StorageDType::F32;
    assert!(profile.validate_inventory(&inventory).is_ok());
    inventory.get_mut(name).unwrap().dtype = StorageDType::Unsupported;
    assert!(profile.validate_inventory(&inventory).is_err());
    inventory.get_mut(name).unwrap().dtype = StorageDType::F16;
    inventory.get_mut(name).unwrap().shape.pop();
    assert!(profile.validate_inventory(&inventory).is_err());
}

#[test]
fn matrix_conversion_preserves_checkpoint_and_transposes_graph_order() {
    let profile =
        CheckpointProfile::from_huggingface_config(&config(false), CheckpointNamespace::Encoder)
            .unwrap();
    let spec = profile
        .weights()
        .iter()
        .find(|spec| spec.graph_name == "layers.0.attn.Wqkv.weight")
        .unwrap();
    let tensor = DecodedTensor {
        shape: spec.checkpoint_shape.clone(),
        values: (0..192).map(|i| i as f32).collect(),
    };
    let actual = spec.graph_values(&tensor).unwrap();
    for row in 0..24 {
        for column in 0..8 {
            assert_eq!(actual[column * 24 + row], tensor.values[row * 8 + column]);
        }
    }
    assert_eq!(tensor.values[1], 1.0);
}

#[test]
fn rejects_missing_wrong_extra_and_nonfinite_encoder_tensors() {
    let profile =
        CheckpointProfile::from_huggingface_config(&config(false), CheckpointNamespace::Encoder)
            .unwrap();
    let good = tensors(&profile);
    assert!(profile.validate_tensors(&good).is_ok());
    let name = "encoder.layers.0.attn.Wqkv.weight";
    let mut missing = good.clone();
    missing.remove(name);
    assert!(profile.validate_tensors(&missing).is_err());
    let mut wrong = good.clone();
    wrong.get_mut(name).unwrap().shape.swap(0, 1);
    assert!(profile.validate_tensors(&wrong).is_err());
    let mut short = good.clone();
    short.get_mut(name).unwrap().values.pop();
    assert!(profile.validate_tensors(&short).is_err());
    let mut nonfinite = good.clone();
    nonfinite.get_mut(name).unwrap().values[0] = f32::NAN;
    assert!(profile.into_reference(nonfinite).is_err());
    let mut extra = good.clone();
    extra.insert(
        "encoder.layers.0.attn.Wqkv.bias".into(),
        DecodedTensor {
            shape: vec![1],
            values: vec![1.0],
        },
    );
    assert!(profile.validate_tensors(&extra).is_err());
    let mut head = good;
    head.insert(
        "head.weight".into(),
        DecodedTensor {
            shape: vec![1],
            values: vec![1.0],
        },
    );
    assert!(profile.validate_tensors(&head).is_ok());
}

#[test]
fn config_rejects_unsupported_activation_missing_settings_and_bad_geometry() {
    let good: serde_json::Value = serde_json::from_slice(&config(false)).unwrap();
    for (key, value) in [
        ("hidden_activation", serde_json::json!("gelu_pytorch_tanh")),
        ("vocab_size", serde_json::json!(0)),
        ("intermediate_size", serde_json::json!(0)),
        ("max_position_embeddings", serde_json::json!(0)),
        ("norm_eps", serde_json::json!(0)),
        ("norm_eps", serde_json::json!(1e-100)),
        (
            "local_attention",
            serde_json::json!(u32::MAX as u64 * 2 + 2),
        ),
    ] {
        let mut bad = good.clone();
        bad[key] = value;
        assert!(
            CheckpointProfile::from_huggingface_config(
                &serde_json::to_vec(&bad).unwrap(),
                CheckpointNamespace::Root
            )
            .is_err(),
            "accepted {key}"
        );
    }
    let mut missing = good;
    missing.as_object_mut().unwrap().remove("mlp_bias");
    assert!(
        CheckpointProfile::from_huggingface_config(
            &serde_json::to_vec(&missing).unwrap(),
            CheckpointNamespace::Root
        )
        .is_err()
    );
}

#[test]
fn graph_settings_preserve_checkpoint_f64_values() {
    let mut value: serde_json::Value = serde_json::from_slice(&config(false)).unwrap();
    value["norm_eps"] = serde_json::json!(0.000010000000123);
    value["rope_parameters"]["full_attention"]["rope_theta"] = serde_json::json!(160000.123456789);
    let profile = CheckpointProfile::from_huggingface_config(
        &serde_json::to_vec(&value).unwrap(),
        CheckpointNamespace::Root,
    )
    .unwrap();
    let settings = profile.graph_settings();
    assert_eq!(settings.norm_epsilon, 0.000010000000123);
    assert_eq!(settings.layers[0].rope_theta, 160000.123456789);
}
