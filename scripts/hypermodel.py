#!/usr/bin/env python3
"""
hypermodel.py

Wraps ChessTransformerDecoder (model.py) in a keras_tuner.HyperModel so
Keras Tuner can search over architecture and optimization hyperparameters
instead of us guessing them by hand. All of the ranges/choices being
searched, and the values held fixed, come from config.yaml (see
config.py) -- nothing here is a hardcoded hyperparameter value.

Requires:
    pip install tensorflow keras-tuner pyyaml

Two things are tuned:
  - build(hp): architecture + optimizer hyperparameters, sampled from
    config.yaml's tuner_search_space section via config.sample_hyperparameter().
  - fit(hp, model, ...): batch_size, because it changes the tf.data
    pipeline itself (create_dataset() takes batch_size as an argument),
    not just the model, so it has to be re-sampled and the datasets
    rebuilt inside fit() rather than fixed ahead of time. See
    https://keras.io/guides/keras_tuner/getting_started/#tune-model-training  # noqa: E501

data.seq_len (config.yaml) is used as max_seq_len but is NOT itself
tunable here -- it's baked into the TFRecord shards by build_dataset.py's
--seq-len, so changing it would mean rebuilding the dataset, not just
re-running the search.

Note on the learning-rate schedule during search: ChessTransformerDecoder
does warmup-then-cosine-decay when given total_steps, but during a search
each trial may run for a different number of epochs (especially with
Hyperband), so there's no single "total_steps" that's correct for every
trial. We leave total_steps=None here, which falls back to warmup-then-hold
-- perfectly fine for the short trials a search runs. Once you've picked a
winning configuration, train.py should set total_steps properly for the
real, full-length training run.
"""

import gc

import keras_tuner as kt
import tensorflow as tf

from build_dataset import create_dataset
from config import sample_hyperparameter
from model import ChessTransformerDecoder


class ChessHyperModel(kt.HyperModel):
    def __init__(self, cfg, vocab_size, pad_id, eos_id):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.max_seq_len = cfg["data"]["seq_len"]
        self.train_tfrecord_pattern = cfg["paths"]["train_tfrecord_pattern"]
        self.val_tfrecord_pattern = cfg["paths"]["val_tfrecord_pattern"]

    def build(self, hp):
        # A long-running search builds a fresh model every trial in the
        # SAME process. TensorFlow/Keras don't fully release the previous
        # trial's graph and GPU-side allocations on their own, so that
        # memory accumulates trial over trial until the OS kills the
        # process outright (macOS's jetsam killer does this rather than
        # letting Python raise a normal MemoryError -- it shows up as
        # "zsh: killed", not a traceback). clear_session() releases Keras'
        # backend-held state, and gc.collect() forces Python to actually
        # drop the now-unreferenced previous model/graph objects (TF's
        # internals commonly have reference cycles that the normal
        # refcounting GC won't clean up immediately on its own).
        tf.keras.backend.clear_session()
        gc.collect()

        search_space = self.cfg["tuner_search_space"]

        def sample(name):
            return sample_hyperparameter(hp, search_space, name)

        # d_model/num_heads choices in config.yaml are picked so every
        # combination divides evenly (128/256/384/512 are all multiples
        # of 2, 4, and 8) -- sampled independently, this never lands on
        # an invalid (d_model, num_heads) pair.
        d_model = sample("d_model")
        num_heads = sample("num_heads")
        num_layers = sample("num_layers")
        # dff as a multiple of d_model (2x-4x is the usual range; the
        # original Transformer paper uses 4x) rather than an independent
        # absolute size, so it scales sensibly with whatever d_model gets
        # picked instead of being either tiny or absurdly oversized.
        dff_multiplier = sample("dff_multiplier")
        dropout_rate = sample("dropout_rate")
        attention_dropout_rate = sample("attention_dropout_rate")
        activation = sample("activation")
        tie_embeddings = sample("tie_embeddings")
        learning_rate = sample("learning_rate")
        warmup_steps = sample("warmup_steps")
        weight_decay = sample("weight_decay")
        beta_2 = sample("beta_2")
        label_smoothing = sample("label_smoothing")
        gradient_clip_norm = sample("gradient_clip_norm")

        # Values not being searched (see config.yaml's model_defaults /
        # optimizer_defaults) are held fixed at their configured default.
        fixed = {**self.cfg["model_defaults"], **self.cfg["optimizer_defaults"]}

        model = ChessTransformerDecoder(
            vocab_size=self.vocab_size,
            pad_id=self.pad_id,
            eos_id=self.eos_id,
            max_seq_len=self.max_seq_len,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dff=d_model * dff_multiplier,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            activation=activation,
            layer_norm_epsilon=fixed["layer_norm_epsilon"],
            initializer_range=fixed["initializer_range"],
            tie_embeddings=tie_embeddings,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            total_steps=None,  # see module docstring
            min_lr_ratio=fixed["min_lr_ratio"],
            weight_decay=weight_decay,
            beta_1=fixed["beta_1"],
            beta_2=beta_2,
            adam_epsilon=fixed["adam_epsilon"],
            gradient_clip_norm=gradient_clip_norm,
            label_smoothing=label_smoothing,
            top_k_metrics=tuple(fixed["top_k_metrics"]),
        )
        model.compile_default()
        # Subclassed Keras models don't get real weights from model.build()
        # unless they implement their own build() method -- ours doesn't,
        # since every sub-layer (Embedding, MultiHeadAttention, Dense, ...)
        # already builds itself lazily on first call. Calling model.build()
        # directly here would just flip a "built" flag without actually
        # creating any weights (and Keras warns about exactly that). A real
        # dummy forward pass is what actually builds every sub-layer.
        dummy_input = tf.zeros((1, self.max_seq_len), dtype=tf.int32)
        model(dummy_input, training=False)
        return model

    def fit(self, hp, model, *args, **kwargs):
        # batch_size affects the data pipeline, not just the model, so
        # it's sampled here (from config.yaml's tuner_search_space, same
        # as everything in build()) and used to (re)build the datasets
        # for this trial, rather than being fixed ahead of time by
        # whoever calls tuner.search().
        batch_size = sample_hyperparameter(hp, self.cfg["tuner_search_space"], "batch_size")

        train_ds = create_dataset(
            self.train_tfrecord_pattern,
            seq_len=self.max_seq_len,
            pad_id=self.pad_id,
            batch_size=batch_size,
            shuffle=True,
        )
        val_ds = create_dataset(
            self.val_tfrecord_pattern,
            seq_len=self.max_seq_len,
            pad_id=self.pad_id,
            batch_size=batch_size,
            shuffle=False,
        )

        # Don't pass x/y/validation_data to tuner.search() when using this
        # HyperModel -- the datasets are built here from batch_size instead.
        return model.fit(train_ds, *args, validation_data=val_ds, **kwargs)
