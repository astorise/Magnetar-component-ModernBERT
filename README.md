# magnetar-component-modernbert

ModernBERT encoder component for [Magnetar](https://github.com/astorise/Magnetar).

## Status

The Rust library validates a checkpoint's bidirectional attention layout and
provides an f32 CPU encoder reference using supplied weights. It preserves
global/local layer order, per-kind RoPE theta, physical or explicit position
IDs, padding masks and the first layer's missing attention LayerNorm. Synthetic
full-encoder tests compare the pinned Transformers layers for both bias forms,
including padded queries and local windows. Model loading, graph production
and Component Model exports remain unimplemented; ModernBERT is not yet
runnable in Magnetar.

## Work required

Bidirectional encoder graphs, padding and local-window attention,
checkpoint-specific tokenization and pooling or task heads remain necessary;
mmBERT compatibility requires separate validation. The necessary Magnetar
encoder and decision contracts are tracked in [issue #214](https://github.com/astorise/Magnetar/issues/214).

A real implementation must identify a specific checkpoint and its license, integrate the relevant Magnetar component and runtime contracts, and validate outputs against an independent reference. See the [upstream model or project](https://huggingface.co/answerdotai/ModernBERT-base). Model weights are not included.

This repository contains a model component, not a hardware Provider. It is intended to be checked out by Magnetar at `components/modernbert`.
