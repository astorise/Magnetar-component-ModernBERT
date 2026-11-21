use magnetar_component_modernbert::{
    AttentionKind,
    encoder_reference::{Encoder, EncoderLayer, LayerNorm, Linear},
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

fn pattern(name: &str, count: usize) -> Vec<f32> {
    let seed = name
        .bytes()
        .enumerate()
        .map(|(index, byte)| (index + 1) * byte as usize)
        .sum::<usize>()
        % 97;
    (0..count)
        .map(|index| {
            let mut value = ((index * 17 + seed) % 101) as f32 - 50.0;
            value /= 200.0;
            if name.contains("norm") && name.ends_with(".weight") {
                value += 1.0;
            }
            value
        })
        .collect()
}

fn linear(name: &str, input: usize, output: usize, bias: bool) -> Linear {
    Linear {
        weight: pattern(&format!("{name}.weight"), input * output),
        bias: bias.then(|| pattern(&format!("{name}.bias"), output)),
    }
}

fn norm(name: &str, hidden: usize, bias: bool) -> LayerNorm {
    LayerNorm {
        weight: pattern(&format!("{name}.weight"), hidden),
        bias: bias.then(|| pattern(&format!("{name}.bias"), hidden)),
    }
}

fn encoder(bias: bool) -> Encoder {
    let hidden = 8;
    let intermediate = 12;
    let layers = (0..3)
        .map(|index| {
            let prefix = format!("layers.{index}");
            let local = index == 1;
            EncoderLayer {
                attention: if local {
                    AttentionKind::Local
                } else {
                    AttentionKind::Global
                },
                rope_theta: if local { 10_000.0 } else { 160_000.0 },
                attention_norm: (index != 0)
                    .then(|| norm(&format!("{prefix}.attn_norm"), hidden, bias)),
                qkv: linear(&format!("{prefix}.attn.Wqkv"), hidden, 3 * hidden, bias),
                attention_output: linear(&format!("{prefix}.attn.Wo"), hidden, hidden, bias),
                mlp_norm: norm(&format!("{prefix}.mlp_norm"), hidden, bias),
                mlp_in: linear(&format!("{prefix}.mlp.Wi"), hidden, 2 * intermediate, bias),
                mlp_out: linear(&format!("{prefix}.mlp.Wo"), intermediate, hidden, bias),
            }
        })
        .collect();
    Encoder {
        vocab_size: 24,
        hidden,
        intermediate,
        heads: 2,
        max_positions: 64,
        local_radius: 2,
        epsilon: 1e-5,
        token_embedding: pattern("embeddings.tok_embeddings.weight", 24 * hidden),
        embedding_norm: norm("embeddings.norm", hidden, bias),
        layers,
        final_norm: norm("final_norm", hidden, bias),
    }
}

#[test]
fn full_encoder_matches_pinned_transformers_layers() {
    let cases: Vec<Case> =
        serde_json::from_str(include_str!("fixtures/encoder_reference.json")).unwrap();
    for case in cases {
        let model = encoder(case.bias);
        let mask: Vec<_> = case.mask.iter().map(|&value| value != 0).collect();
        let actual = model
            .forward(&case.ids, &mask, Some(&case.positions))
            .unwrap();
        assert_eq!(actual.len(), case.output.len());
        for (index, (&actual, &expected)) in actual.iter().zip(&case.output).enumerate() {
            assert!(
                (actual - expected).abs() <= 2e-6 + 1e-6 * expected.abs(),
                "output[{index}] {actual} != {expected}"
            );
        }
        if !case.bias {
            assert_eq!(actual, model.forward(&case.ids, &mask, None).unwrap());
        }
    }
}

#[test]
fn encoder_rejects_invalid_tokens_positions_weights_and_empty_local_windows() {
    let mut model = encoder(false);
    assert!(model.forward(&[1, 99], &[true, true], None).is_err());
    assert!(model.forward(&[1, 2], &[true], None).is_err());
    assert!(
        model
            .forward(&[1, 2], &[true, true], Some(&[0, 64]))
            .is_err()
    );
    assert!(
        model
            .forward(
                &[1, 2, 3, 4, 5, 6],
                &[true, false, false, false, false, false],
                None
            )
            .is_err()
    );
    model.layers[0].qkv.weight[0] = f32::NAN;
    assert!(model.forward(&[1, 2], &[true, true], None).is_err());
    let mut bad_layer = encoder(false);
    bad_layer.layers[0].attention_norm = Some(norm("bad", 8, false));
    assert!(bad_layer.forward(&[1, 2], &[true, true], None).is_err());
}
