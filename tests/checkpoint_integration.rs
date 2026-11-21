use magnetar_component_modernbert::{AttentionKind, EncoderPlan};

fn check_published_plan(bytes: &[u8], expected_hidden: usize, expected_layers: usize) {
    let plan = EncoderPlan::from_huggingface_config(bytes).unwrap();
    assert_eq!(plan.hidden_size, expected_hidden);
    assert_eq!(plan.layers.len(), expected_layers);
    assert_eq!(plan.local_attention, 128);
    for (index, layer) in plan.layers.iter().enumerate() {
        let global = index % 3 == 0;
        assert_eq!(layer.has_attention_norm, index != 0, "layer {index}");
        assert_eq!(
            layer.attention,
            if global {
                AttentionKind::Global
            } else {
                AttentionKind::Local
            },
            "layer {index}"
        );
        assert_eq!(layer.rope_theta, if global { 160_000.0 } else { 10_000.0 });
    }
}

#[test]
fn pinned_base_checkpoint_retains_legacy_attention_and_rope_parameters() {
    check_published_plan(
        include_bytes!("fixtures/modernbert-base.config.json"),
        768,
        22,
    );
}

#[test]
fn laya_encoder_checkpoint_retains_explicit_layers_and_nested_rope_parameters() {
    check_published_plan(
        include_bytes!("fixtures/laya-encoder.config.json"),
        1024,
        28,
    );
}

#[test]
fn explicit_layout_and_nested_rope_override_legacy_fields() {
    let config = br#"{
      "model_type":"modernbert","hidden_size":16,"num_attention_heads":2,
      "num_hidden_layers":2,"global_attn_every_n_layers":0,
      "layer_types":["sliding_attention","full_attention"],
      "global_rope_theta":100,"local_rope_theta":200,
      "rope_parameters":{"full_attention":{"rope_type":"default","rope_theta":500},
                         "sliding_attention":{"rope_type":"default","rope_theta":600}}
    }"#;
    let plan = EncoderPlan::from_huggingface_config(config).unwrap();
    assert_eq!(plan.layers[0].attention, AttentionKind::Local);
    assert_eq!(plan.layers[0].rope_theta, 600.0);
    assert!(!plan.layers[0].has_attention_norm);
    assert_eq!(plan.layers[1].attention, AttentionKind::Global);
    assert_eq!(plan.layers[1].rope_theta, 500.0);
    assert!(plan.layers[1].has_attention_norm);
}

#[test]
fn unsupported_and_malformed_attention_layouts_fail_closed() {
    let configs: &[&[u8]] = &[
        br#"{"model_type":"llama","hidden_size":16,"num_attention_heads":2,"num_hidden_layers":2}"#,
        br#"{"model_type":"modernbert","hidden_size":15,"num_attention_heads":2,"num_hidden_layers":2}"#,
        br#"{"model_type":"modernbert","hidden_size":12,"num_attention_heads":4,"num_hidden_layers":2}"#,
        br#"{"model_type":"modernbert","hidden_size":16,"num_attention_heads":2,"num_hidden_layers":0}"#,
        br#"{"model_type":"modernbert","hidden_size":16,"num_attention_heads":2,"num_hidden_layers":4097}"#,
        br#"{"model_type":"modernbert","hidden_size":16,"num_attention_heads":2,"num_hidden_layers":2,"global_attn_every_n_layers":0}"#,
        br#"{"model_type":"modernbert","hidden_size":16,"num_attention_heads":2,"num_hidden_layers":2,"layer_types":["full_attention"]}"#,
        br#"{"model_type":"modernbert","hidden_size":16,"num_attention_heads":2,"num_hidden_layers":1,"layer_types":["causal_attention"]}"#,
        br#"{"model_type":"modernbert","hidden_size":16,"num_attention_heads":2,"num_hidden_layers":1,"local_attention":0}"#,
        br#"{"model_type":"modernbert","hidden_size":16,"num_attention_heads":2,"num_hidden_layers":1,"rope_scaling":{"type":"linear"}}"#,
        br#"{"model_type":"modernbert","hidden_size":16,"num_attention_heads":2,"num_hidden_layers":1,"rope_parameters":{"full_attention":{"rope_type":"linear"}}}"#,
        br#"{"model_type":"modernbert","hidden_size":16,"num_attention_heads":2,"num_hidden_layers":1,"global_rope_theta":0}"#,
        b"not-json",
    ];
    for config in configs {
        assert!(
            EncoderPlan::from_huggingface_config(config).is_err(),
            "invalid config accepted: {}",
            String::from_utf8_lossy(config)
        );
    }
}
