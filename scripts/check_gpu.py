#!/usr/bin/env python3
"""
check_gpu.py

Confirms TensorFlow can actually see and use the M1's GPU through the
tensorflow-metal PluggableDevice, and gives a rough sense of the speedup
over CPU -- both with a generic matmul and with one real training step of
our own model, so "it imports without errors" isn't mistaken for "it's
actually training on the GPU."

Requires:
    pip install tensorflow tensorflow-metal

Usage:
    python check_gpu.py
"""

import time

import tensorflow as tf


def print_devices():
    print(f"TensorFlow version: {tf.__version__}")
    devices = tf.config.list_physical_devices()
    for d in devices:
        # d.name (e.g. "/physical_device:GPU:0") is just TensorFlow's
        # generic, backend-agnostic device identifier -- not a statement
        # that no device was found. get_device_details() is what surfaces
        # the actual hardware name/vendor for a given device, when the
        # backend provides one.
        details = tf.config.experimental.get_device_details(d)
        extra = f" ({details})" if details else ""
        print(f"  {d.device_type}: {d.name}{extra}")
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print(
            "\nNo GPU device found. tensorflow-metal registers itself as a "
            "'GPU' device via TensorFlow's PluggableDevice mechanism, so if "
            "nothing shows up here, either tensorflow-metal isn't installed "
            "in this environment, or it failed to load (check for an import "
            "warning/error above when tensorflow was imported)."
        )
    return bool(gpus)


def benchmark_matmul(device, size=4096, iterations=5):
    with tf.device(device):
        a = tf.random.normal((size, size))
        b = tf.random.normal((size, size))
        # warm-up: first call on a device pays a one-off compilation/
        # dispatch cost that isn't representative of steady-state speed.
        _ = tf.matmul(a, b)
        start = time.perf_counter()
        for _ in range(iterations):
            c = tf.matmul(a, b)
        _ = c.numpy()  # block until the (async-dispatched) op actually finishes
        elapsed = time.perf_counter() - start
    return elapsed / iterations


def benchmark_training_step(device, vocab_size=4210, batch_size=64, seq_len=40):
    from model import ChessTransformerDecoder

    with tf.device(device):
        model = ChessTransformerDecoder(
            vocab_size=vocab_size, pad_id=0, eos_id=1, max_seq_len=seq_len,
            d_model=256, num_layers=6, num_heads=8,
        )
        model.compile_default()
        input_ids = tf.random.uniform((batch_size, seq_len), minval=0, maxval=vocab_size, dtype=tf.int32)
        labels = tf.random.uniform((batch_size, seq_len), minval=0, maxval=vocab_size, dtype=tf.int32)
        sample_weight = tf.ones((batch_size, seq_len), dtype=tf.float32)

        model.train_on_batch(input_ids, labels, sample_weight=sample_weight)  # warm-up (build + trace)
        start = time.perf_counter()
        for _ in range(5):
            model.train_on_batch(input_ids, labels, sample_weight=sample_weight)
        elapsed = time.perf_counter() - start
    return elapsed / 5


def main():
    has_gpu = print_devices()

    print("\n--- matmul benchmark (4096x4096, average of 5) ---")
    cpu_time = benchmark_matmul("/CPU:0")
    print(f"  CPU: {cpu_time * 1000:.1f} ms/iteration")
    if has_gpu:
        gpu_time = benchmark_matmul("/GPU:0")
        print(f"  GPU: {gpu_time * 1000:.1f} ms/iteration ({cpu_time / gpu_time:.1f}x)")

    print("\n--- one training step of ChessTransformerDecoder"
          " (default-size config, batch=64, seq_len=40) ---")
    cpu_step = benchmark_training_step("/CPU:0")
    print(f"  CPU: {cpu_step * 1000:.1f} ms/step")
    if has_gpu:
        gpu_step = benchmark_training_step("/GPU:0")
        print(f"  GPU: {gpu_step * 1000:.1f} ms/step ({cpu_step / gpu_step:.1f}x)")
        print(
            "\nIf the GPU number here isn't clearly faster than CPU, that's "
            "worth knowing before a long training run -- tensorflow-metal "
            "doesn't have optimized kernels for every op, and a model with "
            "an unsupported op anywhere in it can silently fall back to CPU "
            "for that op, eating the benefit. Small models can also be too "
            "small for the GPU's overhead to pay off. Try d_model=512, "
            "num_layers=8 (config.yaml's tuner range covers this) to see if "
            "the gap widens once there's more compute per step."
        )


if __name__ == "__main__":
    main()
