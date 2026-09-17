#!/usr/bin/env python3
"""
metrics.py

Metrics for the move-prediction transformer, used at model.compile() time.

Everything here relies on the sample_weight tensor that create_dataset()
(build_dataset.py) already produces: Keras applies sample_weight to the
loss AND to every compiled metric automatically, so padded positions never
count toward any of these numbers without any extra masking code.

Usage:
    from build_dataset import create_dataset
    from metrics import build_metrics

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=build_metrics(top_k=(1, 5)),
    )
    model.fit(create_dataset(...), validation_data=create_dataset(...), epochs=10)
"""

import tensorflow as tf


class Perplexity(tf.keras.metrics.Metric):
    """exp(mean cross-entropy loss over non-padding positions).

    Perplexity is the standard way to read a language-model loss: it's
    "the effective number of equally-likely choices the model is torn
    between, on average, at each position." At the very start of training
    it should sit close to the vocabulary size (the model is no better
    than guessing uniformly among move tokens); a model that has actually
    learned chess patterns should drive it down into the single/low-double
    digits. It carries the same information as the loss but is easier to
    reason about across epochs and compare against a "random guessing"
    baseline.
    """

    def __init__(self, name="perplexity", **kwargs):
        super().__init__(name=name, **kwargs)
        self._loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        self.total_loss = self.add_weight(name="total_loss", initializer="zeros")
        self.total_count = self.add_weight(name="total_count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        per_token_loss = self._loss_fn(y_true, y_pred)  # shape (batch, seq_len)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, per_token_loss.dtype)
            per_token_loss = per_token_loss * sample_weight
            count = tf.reduce_sum(sample_weight)
        else:
            count = tf.cast(tf.size(per_token_loss), per_token_loss.dtype)
        self.total_loss.assign_add(tf.reduce_sum(per_token_loss))
        self.total_count.assign_add(count)

    def result(self):
        mean_loss = tf.math.divide_no_nan(self.total_loss, self.total_count)
        return tf.exp(mean_loss)

    def reset_state(self):
        self.total_loss.assign(0.0)
        self.total_count.assign(0.0)


def build_metrics(top_k=(1, 5)):
    """
    Standard metric set for the move-prediction task:
      - top1_acc: fraction of positions where the model's single most
        likely move exactly matches the move the human actually played.
      - top{k}_acc for k > 1: fraction of positions where the human's move
        is among the model's k most likely predictions. More forgiving
        than top-1 -- useful because a position often has several
        reasonable moves, and the human playing one doesn't make the
        others "wrong".
      - perplexity: see the Perplexity class above.
    """
    metrics = []
    for k in top_k:
        if k == 1:
            metrics.append(tf.keras.metrics.SparseCategoricalAccuracy(name="top1_acc"))
        else:
            metrics.append(tf.keras.metrics.SparseTopKCategoricalAccuracy(k=k, name=f"top{k}_acc"))
    metrics.append(Perplexity())
    return metrics
