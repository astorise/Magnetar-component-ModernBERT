# magnetar-component-modernbert

ModernBERT encoder component for [Magnetar](https://github.com/astorise/Magnetar).

## Status

The Rust library validates a checkpoint's bidirectional attention layout and
provides an f32 CPU encoder reference using supplied weights. It preserves
global/local layer order, per-kind RoPE theta, physical or explicit position
IDs, padding masks and the first layer's missing attention LayerNorm. Synthetic
full-encoder tests compare the pinned Transformers layers for both bias forms,
including padded queries and local windows. The WebAssembly Component exports
`build-encoder-graph` using Magnetar's 1.12 encoder graph contract. The caller
supplies token IDs, an attention mask, a validated encoder configuration, and
the checkpoint weights (linear weights transposed to `[input, output]`). The
Component builds a bidirectional graph with physical positions starting at
zero and produces final hidden states.

The full production loading and encoder inference paths are still being
integrated, so this is not yet an end-user inference interface. The numerical
integration test in Magnetar executes the Component's graph against synthetic
weights and compares it with the pinned Transformers implementation; it does
not qualify a published checkpoint.

## Work required

Checkpoint ingestion, checkpoint-specific tokenization and pooling or task
heads remain necessary;
mmBERT compatibility requires separate validation. The necessary Magnetar
encoder and decision contracts are tracked in [issue #214](https://github.com/astorise/Magnetar/issues/214).

A real implementation must identify a specific checkpoint and its license, integrate the relevant Magnetar component and runtime contracts, and validate outputs against an independent reference. See the [upstream model or project](https://huggingface.co/answerdotai/ModernBERT-base). Model weights are not included.

This repository contains a model component, not a hardware Provider. It is intended to be checked out by Magnetar at `components/modernbert`.
