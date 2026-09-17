#!/usr/bin/env python3
"""
model.py

The decoder-only transformer itself, plus everything needed to compile it
for training, wrapped in a single class (ChessTransformerDecoder) that is
initialized with every hyperparameter that matters for this kind of
LLM-style model: architecture size (layers/heads/dimensions), the
optimization schedule (learning rate, warmup, decay, weight decay), and
regularization (dropout, label smoothing, gradient clipping).

Requires:
    pip install tensorflow

Usage (from a training script):
    from model import ChessTransformerDecoder

    model = ChessTransformerDecoder(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_seq_len=40,       # must match --seq-len used in build_dataset.py
        num_layers=6,
        num_heads=8,
        d_model=256,
        learning_rate=3e-4,
        total_steps=steps_per_epoch * num_epochs,   # optional, enables LR decay
    )
    model.compile_default()
    model(tf.zeros((1, 40), dtype=tf.int32))  # builds every sub-layer via a real forward pass
    model.summary()
    model.fit(train_ds, validation_data=val_ds, epochs=10)

Architecture notes
-------------------
- Moves are embedded, added to a learned positional embedding, and passed
  through `num_layers` pre-norm decoder blocks (LayerNorm -> causal
  self-attention -> residual, LayerNorm -> feed-forward -> residual).
  Pre-norm is the GPT-2/GPT-3 convention and trains more stably at depth
  than the original post-norm Transformer.
- Causal masking is handled by MultiHeadAttention's built-in
  `use_causal_mask=True`, so a position can only attend to itself and
  earlier positions -- no separate mask tensor to build.
- No separate padding mask is needed in attention: build_dataset.py always
  puts padding at the *end* of a window, so under a causal mask real
  moves never attend to padding (padding only ever comes after them).
  Padding positions themselves may attend to real context, which is
  harmless since their predictions are excluded from the loss anyway (via
  the sample_weight that create_dataset() produces).
- The output projection can optionally be tied to the input embedding
  weights (tie_embeddings=True, the default) -- a common technique that
  cuts parameter count noticeably at this vocab size and often mildly
  improves quality, since "how similar are two moves" is a signal the
  model can usefully share between reading and predicting tokens.
"""

from typing import Optional, Sequence

import tensorflow as tf

from metrics import build_metrics


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup to `peak_lr` over `warmup_steps`, then cosine decay to
    `peak_lr * min_lr_ratio` by `total_steps`.

    This is the standard schedule for training Transformers: warmup avoids
    destabilizing the randomly-initialized attention layers with a large
    learning rate before the model has any structure yet, and the cosine
    decay lets it settle into a minimum smoothly instead of overshooting
    late in training.

    If `total_steps` is None, there's no decay phase -- the schedule just
    warms up and then holds at `peak_lr`. That's a reasonable way to start
    (e.g. for a first short training run) before you know how many total
    steps a real run will take; pass `total_steps` once you do.
    """

    def __init__(self, peak_lr: float, warmup_steps: int, total_steps: Optional[int] = None,
                 min_lr_ratio: float = 0.1):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        warmup_lr = self.peak_lr * (step / warmup_steps)

        if self.total_steps is None:
            return tf.minimum(warmup_lr, self.peak_lr)

        total_steps = tf.cast(self.total_steps, tf.float32)
        progress = tf.clip_by_value((step - warmup_steps) / tf.maximum(total_steps - warmup_steps, 1.0), 0.0, 1.0)
        min_lr = self.peak_lr * self.min_lr_ratio
        cosine_lr = min_lr + 0.5 * (self.peak_lr - min_lr) * (1.0 + tf.cos(3.14159265 * progress))

        return tf.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr_ratio": self.min_lr_ratio,
        }


class LinearWarmup(tf.keras.callbacks.Callback):
    """Ramp the optimizer's learning rate linearly from ~0 up to `peak_lr`
    over `warmup_steps` batches, then leave it alone.

    This is the callback-based counterpart to WarmupCosineDecay, used when
    ChessTransformerDecoder is compiled with total_steps=None (see
    compile_default() below): in that mode the optimizer is given a plain,
    *settable* learning rate instead of a LearningRateSchedule object,
    specifically so this callback can drive it during warmup, and
    tf.keras.callbacks.ReduceLROnPlateau can take over afterwards -- once
    validation performance actually stalls, not on a pre-committed step
    count. (A LearningRateSchedule-based optimizer's learning_rate is not
    settable at all -- Keras raises a TypeError if you try -- which is
    exactly why the total_steps-is-not-None / cosine-decay path can't mix
    with ReduceLROnPlateau.)
    """

    def __init__(self, peak_lr: float, warmup_steps: int):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_steps = max(1, warmup_steps)
        self._done = False

    def on_train_batch_begin(self, batch, logs=None):
        if self._done:
            return
        step = int(self.model.optimizer.iterations.numpy())
        if step >= self.warmup_steps:
            self.model.optimizer.learning_rate = self.peak_lr
            self._done = True
            return
        self.model.optimizer.learning_rate = self.peak_lr * (step + 1) / self.warmup_steps


class MoveLoss(tf.keras.losses.Loss):
    """Sparse categorical crossentropy with optional label smoothing.

    tf.keras.losses.SparseCategoricalCrossentropy has no label_smoothing
    argument -- that only exists on the dense/one-hot CategoricalCrossentropy
    loss, since smoothing needs a full probability distribution to smooth
    probability mass into, not a single sparse integer label. When
    label_smoothing > 0 this one-hot-encodes the sparse labels on the fly
    (only in that case -- the common label_smoothing=0 case never pays for
    the one-hot) and defers to categorical_crossentropy's own smoothing.

    Uses the *functional* tf.keras.losses.sparse_categorical_crossentropy /
    categorical_crossentropy (not the Loss-class __call__, which reduces to
    a scalar immediately) so this returns one loss value per sequence
    position, shape (batch, seq_len) -- exactly what lets Keras apply the
    per-position sample_weight from create_dataset() to mask out padding,
    the same way it would for a plain built-in loss.

    reduction="mean_with_sample_weight" (rather than the tf.keras.losses.Loss
    default, "sum_over_batch_size") matters a lot here and is not just a
    style choice: "sum_over_batch_size" divides the summed, masked loss by
    the total number of positions in the batch (batch_size * seq_len) --
    including the padding positions that sample_weight already zeroed out
    in the numerator. Since a meaningful fraction of each window is padding
    (windows near the start of a game have little history yet), that
    silently dilutes the reported loss/val_loss by roughly that padding
    fraction: it was reading ~2.2-2.4 when the true average cross-entropy
    over real (non-padded) positions was actually ~6.4-6.9 (visible by
    comparing against metrics.py's Perplexity, which -- unlike this loss --
    was already dividing by sum(sample_weight) correctly, hence
    perplexity/exp(loss) never matched). "mean_with_sample_weight" divides
    by sum(sample_weight) instead -- i.e. the count of real, non-padded
    positions -- which is the mathematically correct masked mean and now
    matches what Perplexity reports.

    Practical effect of this fix: loss/val_loss will read noticeably HIGHER
    from now on (closer to their true, undiluted magnitude) -- that's not a
    regression, it's the number finally being accurate. It also means the
    raw gradient magnitude the optimizer sees is correspondingly larger
    (by roughly the inverse of that same padding fraction) before
    gradient_clip_norm is applied, so expect --gradient-clip-norm to engage
    more often than before; that's the clipping doing its job, not a sign
    of instability.
    """

    def __init__(self, vocab_size, from_logits=True, label_smoothing=0.0, name="move_loss"):
        super().__init__(name=name, reduction="mean_with_sample_weight")
        self.vocab_size = vocab_size
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        if self.label_smoothing <= 0.0:
            return tf.keras.losses.sparse_categorical_crossentropy(
                y_true, y_pred, from_logits=self.from_logits
            )
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=self.vocab_size)
        return tf.keras.losses.categorical_crossentropy(
            y_true_one_hot, y_pred, from_logits=self.from_logits, label_smoothing=self.label_smoothing
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "from_logits": self.from_logits,
            "label_smoothing": self.label_smoothing,
        })
        return config


class DecoderBlock(tf.keras.layers.Layer):
    """One pre-norm causal-attention decoder block."""

    def __init__(self, d_model, num_heads, dff, dropout_rate, attention_dropout_rate,
                 activation, layer_norm_epsilon, initializer_range, **kwargs):
        super().__init__(**kwargs)
        initializer = tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

        self.attn_norm = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=attention_dropout_rate,
            kernel_initializer=initializer,
        )
        self.attn_dropout = tf.keras.layers.Dropout(dropout_rate)

        self.ffn_norm = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)
        self.ffn_dense_1 = tf.keras.layers.Dense(dff, activation=activation, kernel_initializer=initializer)
        self.ffn_dense_2 = tf.keras.layers.Dense(d_model, kernel_initializer=initializer)
        self.ffn_dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        attn_in = self.attn_norm(x)
        attn_out = self.attention(query=attn_in, value=attn_in, key=attn_in,
                                   use_causal_mask=True, training=training)
        x = x + self.attn_dropout(attn_out, training=training)

        ffn_in = self.ffn_norm(x)
        ffn_out = self.ffn_dense_2(self.ffn_dense_1(ffn_in))
        x = x + self.ffn_dropout(ffn_out, training=training)
        return x


class ChessTransformerDecoder(tf.keras.Model):
    """
    Decoder-only transformer that predicts the next chess move (as a UCI
    token id) from a sequence of previous moves.

    All hyperparameters that matter for training an LLM-style model are
    arguments here rather than scattered across a training script, so a
    whole experiment configuration is just "the arguments this class was
    constructed with."

    Architecture hyperparameters
    -----------------------------
    vocab_size            : number of move tokens (from vocab.json).
    pad_id                : the <PAD> token id (from vocab.json).
    eos_id                : the <EOS> token id (from vocab.json); not used
                             inside the model, kept here for convenience
                             since generation/inference code will want it.
    max_seq_len            : context length -- must match --seq-len used in
                             build_dataset.py, since it sizes the learned
                             positional embedding table.
    d_model                : embedding / hidden dimension.
    num_layers              : number of decoder blocks.
    num_heads              : attention heads per block (must divide d_model).
    dff                    : feed-forward inner dimension. Defaults to
                             4 * d_model if not given (standard ratio).
    dropout_rate           : dropout on embeddings and residual branches.
    attention_dropout_rate : dropout inside the attention softmax.
    activation              : feed-forward activation ("gelu" by default).
    layer_norm_epsilon      : epsilon for all LayerNormalization layers.
    initializer_range       : stddev for the truncated-normal weight init
                             (GPT-style small init keeps early training
                             stable).
    tie_embeddings          : share the input embedding matrix as the
                             output projection (see module docstring).

    Optimization hyperparameters (used by compile_default())
    ----------------------------------------------------------
    learning_rate      : peak learning rate after warmup.
    warmup_steps       : linear warmup length, in optimizer steps.
    total_steps        : total training steps, for cosine decay. Leave as
                          None to skip decay (warmup + hold).
    min_lr_ratio       : decay floor, as a fraction of learning_rate.
    weight_decay       : AdamW weight decay.
    beta_1, beta_2     : Adam moment decay rates (beta_2=0.98 rather than
                          Keras's default 0.999 is the common Transformer
                          setting -- it tracks the second moment estimate
                          faster, which suits the noisier gradients of
                          attention layers).
    adam_epsilon       : Adam's numerical-stability epsilon (1e-9 is the
                          value from the original Transformer paper).
    gradient_clip_norm : global-norm gradient clipping threshold.
    label_smoothing    : label smoothing for the cross-entropy loss (0.0
                          disables it).
    top_k_metrics      : which top-k accuracies to report (see metrics.py).
    """

    def __init__(
        self,
        vocab_size: int,
        pad_id: int = 0,
        eos_id: int = 1,
        max_seq_len: int = 40,
        d_model: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dff: Optional[int] = None,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        activation: str = "gelu",
        layer_norm_epsilon: float = 1e-6,
        initializer_range: float = 0.02,
        tie_embeddings: bool = True,
        learning_rate: float = 3e-4,
        warmup_steps: int = 1000,
        total_steps: Optional[int] = None,
        min_lr_ratio: float = 0.1,
        weight_decay: float = 0.01,
        beta_1: float = 0.9,
        beta_2: float = 0.98,
        adam_epsilon: float = 1e-9,
        gradient_clip_norm: float = 1.0,
        label_smoothing: float = 0.0,
        top_k_metrics: Sequence[int] = (1, 5),
        name: str = "chess_transformer_decoder",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")

        # --- architecture hyperparameters (stored for get_config / reference) ---
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff if dff is not None else 4 * d_model
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.activation = activation
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.tie_embeddings = tie_embeddings

        # --- optimization hyperparameters (used by compile_default) ---
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.weight_decay = weight_decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.adam_epsilon = adam_epsilon
        self.gradient_clip_norm = gradient_clip_norm
        self.label_smoothing = label_smoothing
        self.top_k_metrics = tuple(top_k_metrics)

        initializer = tf.keras.initializers.TruncatedNormal(stddev=initializer_range)

        self.token_embedding = tf.keras.layers.Embedding(
            vocab_size, d_model, embeddings_initializer=initializer, name="token_embedding"
        )
        self.position_embedding = tf.keras.layers.Embedding(
            max_seq_len, d_model, embeddings_initializer=initializer, name="position_embedding"
        )
        self.embedding_dropout = tf.keras.layers.Dropout(dropout_rate)

        self.decoder_blocks = [
            DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                dff=self.dff,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                activation=activation,
                layer_norm_epsilon=layer_norm_epsilon,
                initializer_range=initializer_range,
                name=f"decoder_block_{i}",
            )
            for i in range(num_layers)
        ]
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=layer_norm_epsilon)

        if not tie_embeddings:
            self.output_projection = tf.keras.layers.Dense(
                vocab_size, kernel_initializer=initializer, name="output_projection"
            )
        else:
            self.output_projection = None
            self.output_bias = self.add_weight(
                name="output_bias", shape=(vocab_size,), initializer="zeros", trainable=True
            )

        self._embed_scale = tf.math.sqrt(tf.cast(d_model, tf.float32))

    def call(self, inputs, training=False):
        """inputs: int32 tensor of move-token ids, shape (batch, seq_len).
        Returns logits of shape (batch, seq_len, vocab_size)."""
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(seq_len)

        x = self.token_embedding(inputs) * self._embed_scale
        x = x + self.position_embedding(positions)
        x = self.embedding_dropout(x, training=training)

        for block in self.decoder_blocks:
            x = block(x, training=training)
        x = self.final_layer_norm(x)

        if self.tie_embeddings:
            logits = tf.matmul(x, self.token_embedding.embeddings, transpose_b=True)
            logits = logits + self.output_bias
        else:
            logits = self.output_projection(x)
        return logits

    def compile_default(self):
        """Compile with the optimizer/loss/metrics implied by this
        instance's own hyperparameters -- so `ChessTransformerDecoder(...)`
        followed by `.compile_default()` is a complete, ready-to-train
        configuration without repeating any of these choices in the
        training script.

        Two mutually exclusive LR strategies, chosen by whether
        total_steps was given at construction time:

        - total_steps is a number: warmup then cosine-decay to
          learning_rate * min_lr_ratio by total_steps, baked into the
          optimizer as a LearningRateSchedule (self-contained, no extra
          callback needed, but commits to a step count up front -- good
          when you know exactly how long this run will be).
        - total_steps is None: warmup then hold at learning_rate,
          compiled with a plain *settable* LR so training scripts can add
          LinearWarmup(peak_lr=self.learning_rate,
          warmup_steps=self.warmup_steps) to perform the ramp-up, plus
          tf.keras.callbacks.ReduceLROnPlateau to back the LR off only
          once validation performance genuinely plateaus, rather than on
          a pre-committed schedule. This is the better default when you
          don't want to guess the right epoch count in advance.
        """
        loss = MoveLoss(
            vocab_size=self.vocab_size, from_logits=True, label_smoothing=self.label_smoothing
        )

        if self.total_steps is None:
            # Start below peak_lr -- LinearWarmup overwrites this every
            # training batch until warmup_steps is reached, then leaves it
            # at learning_rate for ReduceLROnPlateau to manage from there.
            initial_lr = self.learning_rate / max(1, self.warmup_steps)
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=initial_lr,
                weight_decay=self.weight_decay,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
                epsilon=self.adam_epsilon,
                clipnorm=self.gradient_clip_norm,
            )
        else:
            schedule = WarmupCosineDecay(
                peak_lr=self.learning_rate,
                warmup_steps=self.warmup_steps,
                total_steps=self.total_steps,
                min_lr_ratio=self.min_lr_ratio,
            )
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=schedule,
                weight_decay=self.weight_decay,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
                epsilon=self.adam_epsilon,
                clipnorm=self.gradient_clip_norm,
            )

        # weighted_metrics (not metrics=) is deliberate: the dataset yields
        # (input_ids, labels, sample_weight) triples where sample_weight
        # masks out padding. Keras only forwards that mask into a metric's
        # update_state() if the metric is registered as a *weighted*
        # metric -- plain `metrics=` entries always get sample_weight=None,
        # silently averaging over every position, padding included. Since
        # padding positions are never trained on (the loss masks them out
        # too), the model's output there is unconstrained noise that drifts
        # epoch to epoch -- if a metric averages that in, it inherits that
        # drift on top of its real, stable signal from the non-pad
        # positions. Keeping every metric here weighted keeps it reading
        # only the positions that were actually supervised, consistent
        # with the (already masked) loss/val_loss.
        self.compile(optimizer=optimizer, loss=loss, weighted_metrics=build_metrics(top_k=self.top_k_metrics))

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "pad_id": self.pad_id,
            "eos_id": self.eos_id,
            "max_seq_len": self.max_seq_len,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
            "attention_dropout_rate": self.attention_dropout_rate,
            "activation": self.activation,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "initializer_range": self.initializer_range,
            "tie_embeddings": self.tie_embeddings,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "weight_decay": self.weight_decay,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "adam_epsilon": self.adam_epsilon,
            "gradient_clip_norm": self.gradient_clip_norm,
            "label_smoothing": self.label_smoothing,
            "top_k_metrics": self.top_k_metrics,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def from_vocab_file(vocab_path, **kwargs) -> ChessTransformerDecoder:
    """Convenience constructor: read vocab_size/pad_id/eos_id out of the
    vocab.json produced by build_vocab.py and build the model from it, so
    the training script doesn't have to unpack that file itself."""
    import json
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return ChessTransformerDecoder(
        vocab_size=vocab["vocab_size"],
        pad_id=vocab["pad_id"],
        eos_id=vocab["eos_id"],
        **kwargs,
    )


def from_yaml_config(cfg, vocab_path, total_steps=None, **overrides) -> ChessTransformerDecoder:
    """Build a ChessTransformerDecoder entirely from config.yaml's
    model_defaults / optimizer_defaults sections plus vocab.json -- the
    config-driven equivalent of from_vocab_file() above, for a normal
    (non-tuned) training run rather than a Keras Tuner trial.

    Pass total_steps once you know it (steps_per_epoch * num_epochs) to
    get a proper warmup-then-cosine-decay schedule instead of the
    warmup-then-hold fallback. Anything in **overrides takes precedence
    over the config file, e.g. to try one hand-picked change without
    editing config.yaml.
    """
    import json

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    params = dict(cfg["model_defaults"])
    params.update(cfg["optimizer_defaults"])
    dff_multiplier = params.pop("dff_multiplier")
    params["dff"] = params["d_model"] * dff_multiplier
    params["top_k_metrics"] = tuple(params["top_k_metrics"])
    params.pop("batch_size", None)  # not a model constructor arg -- used for the dataset instead

    params.update(
        vocab_size=vocab["vocab_size"],
        pad_id=vocab["pad_id"],
        eos_id=vocab["eos_id"],
        max_seq_len=cfg["data"]["seq_len"],
        total_steps=total_steps,
    )
    params.update(overrides)
    return ChessTransformerDecoder(**params)


if __name__ == "__main__":
    # Quick manual smoke test with a tiny config and random input ids:
    #     python model.py
    tf.random.set_seed(0)
    tiny_model = ChessTransformerDecoder(
        vocab_size=4210, pad_id=0, eos_id=1, max_seq_len=40,
        d_model=32, num_layers=2, num_heads=4, dff=64,
    )
    tiny_model.compile_default()
    dummy_input = tf.random.uniform((2, 40), minval=0, maxval=4210, dtype=tf.int32)
    output_logits = tiny_model(dummy_input, training=False)
    print("input shape:", dummy_input.shape)
    print("output logits shape:", output_logits.shape)  # expect (2, 40, 4210)
    tiny_model.summary()
