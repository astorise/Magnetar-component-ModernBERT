# ModernBERT configuration fixtures

The JSON files are unmodified public checkpoint configurations; no weights
are included. Both upstream model cards declare Apache-2.0.

- `modernbert-base.config.json`: [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base/blob/8949b909ec900327062f0ebf497f51aef5e6f0c8/config.json), revision `8949b909ec900327062f0ebf497f51aef5e6f0c8`, SHA-256 `1609d59e627c33eaed524b4f01e546d42e84190a079a5a5ded84b212c41c324f`.
- `laya-encoder.config.json`: [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya/blob/fd4f9dbcf4fde222fd43174bdd4c3b08b789e4e1/encoder/config.json), revision `fd4f9dbcf4fde222fd43174bdd4c3b08b789e4e1`, SHA-256 `bf3ab80598fdccf414855a2ce80f22859e4492d06ca8a62ddd1cfb63972f8979`.

`encoder_reference.json` is generated from the embeddings, RoPE and encoder
layer definitions of Transformers 5.23.0 `modeling_modernbert.py` (SHA-256
`29b598c408582bd53c0a48ae34dce2cb008aefc5f6087b03c310b5ce7bb0de9b`),
executed with synthetic deterministic weights and eager attention. It covers
three layers, global and inclusive local windows, padding, explicit position
IDs and both bias forms. No checkpoint weights are included. Regenerate using
PyTorch 2.15:

```sh
python tests/fixtures/generate_encoder_reference.py path/to/modeling_modernbert.py
```

The generator verifies the upstream source digest. These fixtures do not run
either published checkpoint through Magnetar.
