"""
A Multi-Scale RCNN Approach to Bilingual Lyric Emotion Classification
Final Project - Alcaeus Andrea Conner

Pipeline
--------
1. Load bts.csv, clean Genius lyrics, drop Japanese releases.
2. Label every track with a Thayer quadrant (Happy / Sad / Raring / Calm)
   from Spotify valence and energy.
3. Train and compare four models:
     a. TF-IDF + LinearSVC                     (midterm baseline)
     b. TF-IDF + metadata + LinearSVC          (baseline with the same extra features)
     c. Dual-Input 1D CNN                      (midterm prototype)
     d. Dual-Stream Multi-Scale RCNN           (final model: multi-scale Conv1D ->
                                                BiGRU -> additive attention pooling
                                                + metadata fusion, AdamW + cosine decay)
4. Evaluate under two protocols:
     - holdout : stratified 80/20 split, random_state=42 (same split as the midterm)
     - cv      : 5-fold StratifiedGroupKFold grouped by song, so remixes, alternate-
                 language and repackaged versions of one song never sit on both
                 sides of a split
5. Save metrics tables, classification reports, confusion matrices, learning
   curves, a Happy-vs-Raring error analysis and attention-word summaries to results/.

Usage
-----
    python main.py                                  # everything, both protocols
    python main.py --models svc svc_meta            # baselines only (no TensorFlow needed)
    python main.py --protocol cv --epochs 40
    python main.py --fasttext cc.en.300.vec         # use pre-trained FastText vectors
    python main.py --group-by album                 # group CV folds by album instead of song
"""

import argparse
import json
import os
import random
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
SEED = 42
RESULTS_DIR = "results"

# Metadata fed to the fusion branch. Valence and energy are deliberately NOT here:
# the Thayer labels are computed from them, so feeding them to the model lets it read
# the answer off the input (see --include-label-features and the report).
BASE_META_COLS = ["year", "is_korean", "is_english", "is_japanese"]
ACOUSTIC_COLS = [
    "spotify_track_danceability", "spotify_track_tempo", "spotify_track_loudness",
    "spotify_track_acousticness", "spotify_track_speechiness",
    "spotify_track_instrumentalness", "spotify_track_liveness", "spotify_track_mode",
]
LABEL_SOURCE_COLS = ["spotify_track_valence", "spotify_track_energy"]

# Text model settings
MAX_VOCAB = 8000
CNN_SEQ_LEN = 300          # midterm prototype setting
RCNN_SEQ_LEN = 450         # median cleaned song is ~450 tokens
EMBED_DIM = 300            # matches 300-d FastText (used when --fasttext is given)
SCRATCH_EMBED_DIM = 128    # learned-from-scratch embedding size (small data: 220 songs)
GRU_UNITS = 64
THAYER_VALENCE_T = 0.5
THAYER_ENERGY_T = 0.75


def set_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def clean_genius_lyrics(text):
    """Remove Genius structural tags ([Verse], [Chorus]...) and scrape artifacts."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"\d*Embed$", "", text)
    text = re.sub(r"You might also like", " ", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def song_key(title):
    """Normalise a track title so every version of one song shares a key.
    'Dynamite (Tropical Remix)', 'Dynamite (Instrumental)' -> 'dynamite'."""
    t = str(title).lower()
    t = re.split(r"\s+-\s+", t)[0]                 # 'No More Dream - Japanese Ver.'
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", t)
    t = re.sub(r"\b(feat|ft)\..*$", " ", t)
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return JAPANESE_TITLE_ALIASES.get(t, t)


# Japanese releases whose romanised title differs from the Korean original
JAPANESE_TITLE_ALIASES = {
    "shingeki no boudan": "attack on bangtan",
    "hormone sensou": "war of hormone",
}


def assign_thayer_emotion(valence, energy):
    if valence >= THAYER_VALENCE_T and energy >= THAYER_ENERGY_T:
        return "Happy"
    if valence < THAYER_VALENCE_T and energy < THAYER_ENERGY_T:
        return "Sad"
    if valence < THAYER_VALENCE_T and energy >= THAYER_ENERGY_T:
        return "Raring"
    return "Calm"


def load_and_preprocess_data(csv_path="bts.csv", include_japanese=False):
    print("Loading and preprocessing data...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Place bts.csv next to main.py.")

    df = pd.read_csv(csv_path)
    n_raw = len(df)
    df = df.dropna(subset=["lyrics", "spotify_track_valence", "spotify_track_energy"]).copy()

    # Drop Japanese releases (language column + kana check on the raw text) unless asked
    if not include_japanese:
        df = df[~df["lang"].fillna("").str.lower().str.contains("jpn|japanese")]
        df = df[~df["lyrics"].str.contains(r"[぀-ゟ゠-ヿ]", regex=True, na=False)]

    df["clean_lyrics"] = df["lyrics"].apply(clean_genius_lyrics)
    df = df[df["clean_lyrics"].str.strip() != ""].copy()

    # Language flags. The lyrics column holds English text for every track (Korean songs
    # are stored as English translations), so a Hangul regex on the text is always 0.
    # The original release language comes from the `lang` column instead.
    lang = df["lang"].fillna("").str.upper()
    has_hangul = df["lyrics"].str.contains(r"[ㄱ-ㆎ가-힣]", regex=True)
    df["is_korean"] = ((lang == "KOR") | has_hangul).astype(float)
    df["is_english"] = (lang == "ENG").astype(float)
    df["is_japanese"] = (lang == "JPN").astype(float)

    df["year"] = pd.to_datetime(df["album_rd"], errors="coerce").dt.year.astype(float)
    df["year"] = df["year"].fillna(df["year"].median())

    for c in ACOUSTIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    df["tag"] = [assign_thayer_emotion(v, e) for v, e in
                 zip(df["spotify_track_valence"], df["spotify_track_energy"])]
    df["song_key"] = df["eng_track_title"].fillna(df["track_title"]).apply(song_key)
    df = df.reset_index(drop=True)

    print(f"Raw rows: {n_raw} | after filtering: {len(df)} | unique songs: {df['song_key'].nunique()}"
          f" | albums: {df['album_title'].nunique()}")
    return df


# --------------------------------------------------------------------------------------
# Text utilities (pure Python so they work with both Keras 2 and Keras 3)
# --------------------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[a-z0-9']+|[가-힣]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class SimpleTokenizer:
    """Word-level tokenizer: 0 = padding, 1 = out-of-vocabulary."""

    def __init__(self, num_words=MAX_VOCAB):
        self.num_words = num_words
        self.word_index = {}

    def fit(self, texts):
        counts = Counter(tok for t in texts for tok in tokenize(t))
        vocab = [w for w, _ in counts.most_common(self.num_words - 2)]
        self.word_index = {w: i + 2 for i, w in enumerate(vocab)}
        self.index_word = {i: w for w, i in self.word_index.items()}
        self.index_word[1] = "<OOV>"
        return self

    def encode(self, texts, maxlen):
        out = np.zeros((len(texts), maxlen), dtype="int32")
        for r, t in enumerate(texts):
            ids = [self.word_index.get(tok, 1) for tok in tokenize(t)][:maxlen]
            out[r, :len(ids)] = ids
        return out

    @property
    def vocab_size(self):
        return len(self.word_index) + 2


def load_fasttext_matrix(path, tokenizer, dim=EMBED_DIM):
    """Build an embedding matrix from a FastText .vec file (reads only needed words)."""
    rng = np.random.default_rng(SEED)
    matrix = rng.normal(0, 0.1, (tokenizer.vocab_size, dim)).astype("float32")
    matrix[0] = 0.0
    found = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        first = f.readline().split()
        if len(first) != 2:                      # no header line
            f.seek(0)
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            idx = tokenizer.word_index.get(word)
            if idx is not None and len(parts) == dim + 1:
                matrix[idx] = np.asarray(parts[1:], dtype="float32")
                found += 1
    print(f"FastText: matched {found}/{tokenizer.vocab_size - 2} vocabulary words")
    return matrix


# --------------------------------------------------------------------------------------
# Metrics and plots
# --------------------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}


def plot_confusion_matrix(y_true, y_pred, classes, title, path, cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    plt.figure(figsize=(6.5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_learning_curves(history, title, path):
    h = history.history
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(h["accuracy"], label="Train")
    ax[0].plot(h["val_accuracy"], label="Validation")
    ax[0].set_title(f"{title} - accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].legend()
    ax[1].plot(h["loss"], label="Train")
    ax[1].plot(h["val_loss"], label="Validation")
    ax[1].set_title(f"{title} - loss")
    ax[1].set_xlabel("Epoch")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_label_space(df, path):
    colors = {"Happy": "#E69F00", "Raring": "#D55E00", "Sad": "#0072B2", "Calm": "#009E73"}
    plt.figure(figsize=(6, 5))
    for tag, g in df.groupby("tag"):
        plt.scatter(g["spotify_track_valence"], g["spotify_track_energy"], s=18,
                    alpha=0.75, label=f"{tag} (n={len(g)})", color=colors.get(tag))
    plt.axvline(THAYER_VALENCE_T, color="grey", ls="--", lw=1)
    plt.axhline(THAYER_ENERGY_T, color="grey", ls="--", lw=1)
    plt.xlabel("Spotify valence")
    plt.ylabel("Spotify energy")
    plt.title("Thayer quadrant labels")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
def run_svc(train_df, test_df, y_train, meta_train=None, meta_test=None):
    """TF-IDF (1-2 grams, 3000 features) + LinearSVC; optional scaled metadata appended."""
    t0 = time.time()
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    Xtr = tfidf.fit_transform(train_df["clean_lyrics"])
    Xte = tfidf.transform(test_df["clean_lyrics"])
    if meta_train is not None:
        Xtr = hstack([Xtr, csr_matrix(meta_train)]).tocsr()
        Xte = hstack([Xte, csr_matrix(meta_test)]).tocsr()
    clf = LinearSVC(class_weight="balanced", C=0.5, max_iter=5000, random_state=SEED)
    clf.fit(Xtr, y_train)
    train_time = time.time() - t0
    t1 = time.time()
    scores = clf.decision_function(Xte)
    pred = scores.argmax(1)
    return pred, train_time, time.time() - t1, {"scores": scores}


def _attention_layer_class():
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    class AdditiveAttentionPooling(layers.Layer):
        """Bahdanau-style additive self-attention pooling over time.
        score_t = v^T tanh(W h_t + b); padded positions (token id 0) are masked out.
        Returns the attention-weighted context vector and the weights."""

        def __init__(self, units=64, l2=1e-4, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.l2 = l2

        def build(self, input_shape):
            d = int(input_shape[0][-1])
            reg = regularizers.l2(self.l2)
            self.W = self.add_weight(name="att_W", shape=(d, self.units),
                                     initializer="glorot_uniform", regularizer=reg)
            self.b = self.add_weight(name="att_b", shape=(self.units,), initializer="zeros")
            self.v = self.add_weight(name="att_v", shape=(self.units, 1),
                                     initializer="glorot_uniform", regularizer=reg)
            super().build(input_shape)

        def call(self, inputs):
            h, token_ids = inputs
            u = tf.tanh(tf.tensordot(h, self.W, axes=1) + self.b)      # (B, T, units)
            score = tf.squeeze(tf.tensordot(u, self.v, axes=1), -1)    # (B, T)
            mask = tf.cast(tf.not_equal(token_ids, 0), score.dtype)
            score = score + (1.0 - mask) * -1e9
            alpha = tf.nn.softmax(score, axis=1)                       # (B, T)
            context = tf.reduce_sum(h * tf.expand_dims(alpha, -1), axis=1)
            return context, alpha

        def get_config(self):
            cfg = super().get_config()
            cfg.update({"units": self.units, "l2": self.l2})
            return cfg

    return AdditiveAttentionPooling


def build_cnn(vocab_size, n_meta, n_classes, seq_len=CNN_SEQ_LEN):
    """Midterm prototype: Dual-Input 1D CNN (single kernel size, global max pooling)."""
    from tensorflow.keras import layers, Model, regularizers
    text_in = layers.Input(shape=(seq_len,), dtype="int32", name="text_input")
    x = layers.Embedding(vocab_size, 64)(text_in)
    x = layers.SpatialDropout1D(0.3)(x)
    x = layers.Conv1D(64, 5, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(x)
    meta_in = layers.Input(shape=(n_meta,), name="meta_input")
    m = layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(meta_in)
    z = layers.Concatenate()([x, m])
    z = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-3))(z)
    z = layers.Dropout(0.5)(z)
    out = layers.Dense(n_classes, activation="softmax")(z)
    model = Model([text_in, meta_in], out, name="dual_input_1d_cnn")
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model, None


def build_rcnn(vocab_size, n_meta, n_classes, steps_per_epoch, epochs,
               embedding_matrix=None, seq_len=RCNN_SEQ_LEN, l2=1e-4):
    """Final model: Dual-Stream Multi-Scale RCNN.

    Text stream : Embedding(300 FastText / 128 learned) -> SpatialDropout(0.3) -> parallel
                  Conv1D k=3/5/7 (multi-scale n-grams) -> concat -> BiGRU(64, recurrent_dropout=0.2)
                  -> additive attention pooling
    Meta stream : Dense(32) over scaled year / language flags / acoustic features
    Fusion      : concat -> Dense(64) -> Dropout(0.5) -> softmax
    Optimiser   : AdamW with cosine-annealed learning rate
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model, regularizers
    reg = regularizers.l2(l2)
    Attention = _attention_layer_class()

    text_in = layers.Input(shape=(seq_len,), dtype="int32", name="text_input")
    if embedding_matrix is not None:
        emb = layers.Embedding(vocab_size, embedding_matrix.shape[1], trainable=False,
                               embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                               name="fasttext_embedding")
    else:
        emb = layers.Embedding(vocab_size, SCRATCH_EMBED_DIM, name="embedding")
    x = emb(text_in)
    x = layers.SpatialDropout1D(0.3)(x)

    branches = [layers.Conv1D(64, k, padding="same", activation="relu", kernel_regularizer=reg,
                              name=f"conv_k{k}")(x) for k in (3, 5, 7)]
    x = layers.Concatenate(name="multi_scale_concat")(branches)
    x = layers.Bidirectional(layers.GRU(GRU_UNITS, return_sequences=True, recurrent_dropout=0.2,
                                        kernel_regularizer=reg), name="bigru")(x)
    context, alpha = Attention(units=64, l2=l2, name="additive_attention")([x, text_in])

    meta_in = layers.Input(shape=(n_meta,), name="meta_input")
    m = layers.Dense(32, activation="relu", kernel_regularizer=reg)(meta_in)

    z = layers.Concatenate(name="fusion")([context, m])
    z = layers.Dense(64, activation="relu", kernel_regularizer=reg)(z)
    z = layers.Dropout(0.5)(z)
    out = layers.Dense(n_classes, activation="softmax", name="emotion")(z)

    model = Model([text_in, meta_in], out, name="dual_stream_multiscale_rcnn")
    attention_model = Model(text_in, alpha, name="attention_weights")

    schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=2e-3, decay_steps=max(1, steps_per_epoch * epochs), alpha=0.05)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=schedule, weight_decay=1e-4)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, attention_model


def validation_split(train_df, y_train, val_mode):
    idx = np.arange(len(train_df))
    if val_mode == "group":
        try:
            sgk = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
            return next(sgk.split(idx, y_train, train_df["song_key"].values))
        except ValueError:
            pass
    return train_test_split(idx, test_size=0.15, random_state=SEED, stratify=y_train)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def run_hybrid(deep_extra, train_df, test_df, y_train, meta_train, meta_test):
    """Late-fusion hybrid: blend the RCNN's softmax with the TF-IDF + Meta LinearSVC.
    The blend weight w (RCNN share) is chosen on the RCNN's own validation split,
    with an SVC fitted on the remaining training rows, so the test set is never used
    for selection. The SVC is then refitted on the full training portion."""
    tr_idx, va_idx = deep_extra["tr_idx"], deep_extra["va_idx"]
    t0 = time.time()
    _, _, _, sv = run_svc(train_df.iloc[tr_idx], train_df.iloc[va_idx], y_train[tr_idx],
                          meta_train[tr_idx], meta_train[va_idx])
    svc_val = softmax(sv["scores"])
    best_w, best_f1 = 0.0, -1.0
    for w in np.round(np.linspace(0, 1, 11), 2):
        p = (w * deep_extra["val_probs"] + (1 - w) * svc_val).argmax(1)
        f1 = precision_recall_fscore_support(y_train[va_idx], p, average="macro", zero_division=0)[2]
        if f1 > best_f1 + 1e-9:
            best_w, best_f1 = float(w), f1
    _, _, inf_t, st = run_svc(train_df, test_df, y_train, meta_train, meta_test)
    probs = best_w * deep_extra["probs"] + (1 - best_w) * softmax(st["scores"])
    return probs.argmax(1), time.time() - t0, inf_t, {"blend_weight_rcnn": best_w}


def run_deep(kind, train_df, test_df, y_train, meta_train, meta_test, n_classes, args, tag,
             val_mode="group"):
    """Train the CNN or RCNN. A validation set for early stopping is carved out of the
    training data only (the midterm script validated on the test set). The validation
    split mirrors the protocol: random for the holdout split, song-grouped for grouped CV."""
    import tensorflow as tf
    set_seeds(SEED)
    seq_len = CNN_SEQ_LEN if kind == "cnn" else RCNN_SEQ_LEN

    # Validation split inside the training portion, grouped by song
    tr_idx, va_idx = validation_split(train_df, y_train, val_mode)

    texts = train_df["clean_lyrics"].values
    tok = SimpleTokenizer(MAX_VOCAB).fit(texts[tr_idx])
    X_tr = tok.encode(texts[tr_idx], seq_len)
    X_va = tok.encode(texts[va_idx], seq_len)
    X_te = tok.encode(test_df["clean_lyrics"].values, seq_len)

    emb_matrix = None
    if kind == "rcnn" and args.fasttext:
        emb_matrix = load_fasttext_matrix(args.fasttext, tok)

    batch = 16
    steps = int(np.ceil(len(tr_idx) / batch))
    if kind == "cnn":
        model, att_model = build_cnn(tok.vocab_size, meta_train.shape[1], n_classes, seq_len)
    else:
        model, att_model = build_rcnn(tok.vocab_size, meta_train.shape[1], n_classes,
                                      steps, args.epochs, emb_matrix, seq_len)

    # Balanced class weights (the baseline uses class_weight='balanced' as well)
    counts = np.bincount(y_train[tr_idx], minlength=n_classes)
    class_weight = {i: len(tr_idx) / (n_classes * max(c, 1)) for i, c in enumerate(counts)}

    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                          restore_best_weights=True, verbose=0)
    t0 = time.time()
    history = model.fit(
        {"text_input": X_tr, "meta_input": meta_train[tr_idx]}, y_train[tr_idx],
        validation_data=({"text_input": X_va, "meta_input": meta_train[va_idx]}, y_train[va_idx]),
        epochs=args.epochs, batch_size=batch, callbacks=[es], class_weight=class_weight,
        verbose=args.verbose)
    train_time = time.time() - t0

    t1 = time.time()
    probs = model.predict({"text_input": X_te, "meta_input": meta_test}, verbose=0)
    inf_time = time.time() - t1
    pred = probs.argmax(1)
    val_probs = model.predict({"text_input": X_va, "meta_input": meta_train[va_idx]}, verbose=0)

    plot_learning_curves(history, MODEL_NAMES[kind], os.path.join(RESULTS_DIR, f"learning_curves_{kind}_{tag}.png"))

    extras = {"epochs_run": len(history.history["loss"]), "params": int(model.count_params()),
              "probs": probs, "val_probs": val_probs, "tr_idx": tr_idx, "va_idx": va_idx}
    if att_model is not None:
        alpha = att_model.predict(X_te, verbose=0)
        extras["attention"] = (X_te, alpha, tok)
    return pred, train_time, inf_time, extras


CMAPS = {"svc": "Blues", "svc_meta": "Greens", "cnn": "Purples", "rcnn": "Oranges", "hybrid": "Reds"}

MODEL_NAMES = {
    "svc": "TF-IDF + LinearSVC",
    "svc_meta": "TF-IDF + Meta + LinearSVC",
    "cnn": "Dual-Input 1D CNN",
    "rcnn": "Dual-Stream Multi-Scale RCNN",
    "hybrid": "Hybrid RCNN + TF-IDF/Meta",
}


# --------------------------------------------------------------------------------------
# Error analysis
# --------------------------------------------------------------------------------------
def top_attended_words(X, alpha, tok, rows, k=3):
    words = Counter()
    for r in rows:
        valid = X[r] > 1                       # skip padding and OOV
        if not valid.any():
            continue
        a = np.where(valid, alpha[r], -1)
        for pos in np.argsort(a)[::-1][:k]:
            words[tok.index_word.get(int(X[r, pos]), "?")] += 1
    return words


def error_analysis(test_df, y_true, preds, classes, attention, path_prefix):
    """Misclassified tracks + the Happy/Raring boundary + attention words per class."""
    out = test_df[["track_title", "album_title", "spotify_track_valence",
                   "spotify_track_energy", "tag"]].copy()
    for name, p in preds.items():
        out[f"pred_{name}"] = [classes[i] for i in p]
    out.to_csv(f"{path_prefix}_predictions.csv", index=False)

    lines = []
    ci = {c: i for i, c in enumerate(classes)}
    for name, p in preds.items():
        cm = confusion_matrix(y_true, p, labels=range(len(classes)))
        if "Happy" in ci and "Raring" in ci:
            h, r = ci["Happy"], ci["Raring"]
            lines.append(f"{MODEL_NAMES[name]}: Happy->Raring {cm[h, r]}, Raring->Happy {cm[r, h]}, "
                         f"total errors {int((p != y_true).sum())} / {len(y_true)}")
        # Distance of misclassified tracks to the Thayer boundary
        wrong = p != y_true
        if wrong.any():
            dv = np.abs(test_df["spotify_track_valence"].values - THAYER_VALENCE_T)
            de = np.abs(test_df["spotify_track_energy"].values - THAYER_ENERGY_T)
            dist = np.minimum(dv, de)
            lines.append(f"    median distance to quadrant boundary: wrong {np.median(dist[wrong]):.3f}"
                         f" vs correct {np.median(dist[~wrong]) if (~wrong).any() else float('nan'):.3f}")

    if attention is not None:
        X, alpha, tok = attention
        lines.append("\nMost-attended words (top-3 per song, RCNN), by true class:")
        for c in classes:
            rows = np.where(y_true == ci[c])[0]
            common = ", ".join(f"{w} ({n})" for w, n in top_attended_words(X, alpha, tok, rows).most_common(10))
            lines.append(f"  {c}: {common}")

    with open(f"{path_prefix}_error_analysis.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# --------------------------------------------------------------------------------------
# Experiment runners
# --------------------------------------------------------------------------------------
def prepare_meta(train_df, test_df, meta_cols):
    scaler = StandardScaler()
    return (scaler.fit_transform(train_df[meta_cols]).astype("float32"),
            scaler.transform(test_df[meta_cols]).astype("float32"))


def run_models(train_df, test_df, le, meta_cols, args, tag):
    y_train = le.transform(train_df["tag"])
    y_test = le.transform(test_df["tag"])
    meta_tr, meta_te = prepare_meta(train_df, test_df, meta_cols)
    n_classes = len(le.classes_)

    val_mode = "random" if tag == "holdout" else "group"
    rows, preds, attention, deep_out = [], {}, None, {}
    for name in args.models:
        print(f"  [{tag}] training {MODEL_NAMES[name]} ...")
        if name == "svc":
            pred, tt, it, extra = run_svc(train_df, test_df, y_train)
        elif name == "svc_meta":
            pred, tt, it, extra = run_svc(train_df, test_df, y_train, meta_tr, meta_te)
        elif name == "hybrid":
            if "rcnn" not in deep_out:
                raise ValueError("--models must list rcnn before hybrid")
            pred, tt, it, extra = run_hybrid(deep_out["rcnn"], train_df, test_df, y_train, meta_tr, meta_te)
            tt += rows[[r["Model"] for r in rows].index(MODEL_NAMES["rcnn"])]["Train Time (s)"]
            print(f"    blend weight on RCNN (chosen on validation): {extra['blend_weight_rcnn']}")
        else:
            pred, tt, it, extra = run_deep(name, train_df, test_df, y_train, meta_tr, meta_te,
                                           n_classes, args, tag, val_mode)
            if extra and "attention" in extra:
                attention = extra.pop("attention")
            deep_out[name] = extra
        m = compute_metrics(y_test, pred)
        m.update({"Model": MODEL_NAMES[name], "Train Time (s)": tt, "Inference Time (s)": it})
        if extra:
            m.update({k: v for k, v in extra.items() if np.isscalar(v)})
        rows.append(m)
        preds[name] = pred
    return rows, preds, y_test, attention


def holdout_protocol(df, le, meta_cols, args):
    print("\n=== Protocol 1: stratified 80/20 holdout (midterm split, random_state=42) ===")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["tag"])
    overlap = test_df["song_key"].isin(set(train_df["song_key"])).mean()
    print(f"  test tracks that have another version of the same song in train: {overlap:.1%}")

    rows, preds, y_test, attention = run_models(train_df, test_df, le, meta_cols, args, "holdout")
    reports = []
    for name, p in preds.items():
        rep = classification_report(y_test, p, target_names=le.classes_, digits=4, zero_division=0)
        reports.append(f"== {MODEL_NAMES[name]} ==\n{rep}")
        cmap = CMAPS[name]
        plot_confusion_matrix(y_test, p, le.classes_, f"{MODEL_NAMES[name]} (holdout)",
                              os.path.join(RESULTS_DIR, f"cm_{name}_holdout.png"), cmap)
    with open(os.path.join(RESULTS_DIR, "classification_reports_holdout.txt"), "w") as f:
        f.write("\n".join(reports))
    print("\n".join(reports))
    error_analysis(test_df, y_test, preds, list(le.classes_), attention,
                   os.path.join(RESULTS_DIR, "holdout"))
    res = pd.DataFrame(rows)
    res.insert(1, "Protocol", "holdout 80/20")
    return res, overlap


def cv_protocol(df, le, meta_cols, args):
    group_col = "song_key" if args.group_by == "song" else "album_title"
    print(f"\n=== Protocol 2: 5-fold StratifiedGroupKFold grouped by {args.group_by} ===")
    sgk = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    all_rows = []
    pooled = {n: np.zeros(len(df), dtype=int) for n in args.models}
    y_all = le.transform(df["tag"])
    for fold, (tr, te) in enumerate(sgk.split(df, df["tag"], df[group_col]), 1):
        print(f" Fold {fold}: train={len(tr)} test={len(te)}")
        rows, preds, _, _ = run_models(df.iloc[tr], df.iloc[te], le, meta_cols, args, f"cv_fold{fold}")
        for r in rows:
            r["Fold"] = fold
        all_rows += rows
        for n, p in preds.items():
            pooled[n][te] = p

    per_fold = pd.DataFrame(all_rows)
    per_fold.to_csv(os.path.join(RESULTS_DIR, f"metrics_cv_per_fold_{args.group_by}.csv"), index=False)

    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "Train Time (s)", "Inference Time (s)"]
    agg = per_fold.groupby("Model", sort=False)[metric_cols].agg(["mean", "std"])
    summary = pd.DataFrame({"Model": agg.index})
    for c in metric_cols:
        summary[c] = agg[(c, "mean")].values
        summary[c + " std"] = agg[(c, "std")].values
    summary.insert(1, "Protocol", f"5-fold grouped by {args.group_by}")

    for n in args.models:
        plot_confusion_matrix(y_all, pooled[n], le.classes_, f"{MODEL_NAMES[n]} (CV, pooled)",
                              os.path.join(RESULTS_DIR, f"cm_{n}_cv_{args.group_by}.png"),
                              CMAPS[n])
    error_analysis(df, y_all, pooled, list(le.classes_), None,
                   os.path.join(RESULTS_DIR, f"cv_{args.group_by}"))
    return summary


def label_leakage_check(df, le):
    """Shows why valence/energy are excluded from the inputs."""
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    y = le.transform(df["tag"])
    accs = []
    for tr, te in sgk.split(df, y, df["song_key"]):
        s = StandardScaler().fit(df.iloc[tr][LABEL_SOURCE_COLS])
        clf = LinearSVC(class_weight="balanced", max_iter=5000, random_state=SEED)
        clf.fit(s.transform(df.iloc[tr][LABEL_SOURCE_COLS]), y[tr])
        accs.append(accuracy_score(y[te], clf.predict(s.transform(df.iloc[te][LABEL_SOURCE_COLS]))))
    return float(np.mean(accs))


# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Bilingual BTS lyric emotion classification")
    p.add_argument("--data", default="bts.csv")
    p.add_argument("--models", nargs="+", default=["svc", "svc_meta", "cnn", "rcnn", "hybrid"],
                   choices=list(MODEL_NAMES))
    p.add_argument("--protocol", choices=["holdout", "cv", "both"], default="both")
    p.add_argument("--group-by", choices=["song", "album"], default="song")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--fasttext", default=None, help="path to a 300-d FastText .vec file")
    p.add_argument("--include-label-features", action="store_true",
                   help="also feed valence/energy (leaks the label; for demonstration only)")
    p.add_argument("--include-japanese", action="store_true",
                   help="keep Japanese releases (their lyrics are English translations too)")
    p.add_argument("--no-acoustic", action="store_true", help="metadata = year + language flags only")
    p.add_argument("--verbose", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    set_seeds(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_and_preprocess_data(args.data, include_japanese=args.include_japanese)
    print("Class distribution:\n", df["tag"].value_counts().to_string())
    plot_label_space(df, os.path.join(RESULTS_DIR, "thayer_label_space.png"))

    meta_cols = BASE_META_COLS + ([] if args.no_acoustic else ACOUSTIC_COLS)
    if args.include_label_features:
        meta_cols += LABEL_SOURCE_COLS
        print("WARNING: valence/energy included as inputs - the labels are derived from them.")
    meta_cols = [c for c in meta_cols if df[c].nunique() > 1]
    print("Metadata features:", meta_cols)

    le = LabelEncoder().fit(df["tag"])
    tables = []
    info = {"n_tracks": len(df), "n_songs": int(df["song_key"].nunique()),
            "class_counts": df["tag"].value_counts().to_dict(), "meta_cols": meta_cols,
            "majority_baseline_acc": float(df["tag"].value_counts(normalize=True).max()),
            "valence_energy_only_cv_acc": label_leakage_check(df, le)}

    if args.protocol in ("holdout", "both"):
        res, overlap = holdout_protocol(df, le, meta_cols, args)
        info["holdout_song_overlap"] = float(overlap)
        tables.append(res)
    if args.protocol in ("cv", "both"):
        tables.append(cv_protocol(df, le, meta_cols, args))

    summary = pd.concat(tables, ignore_index=True)
    summary.to_csv(os.path.join(RESULTS_DIR, "metrics_summary.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    show = ["Model", "Protocol", "Accuracy", "Precision", "Recall", "F1-Score",
            "Train Time (s)", "Inference Time (s)"]
    print("\n" + "=" * 100)
    print("MODEL PERFORMANCE, TIMING, AND ERROR METRICS SUMMARY")
    print("=" * 100)
    print(summary[show].round(4).to_string(index=False))
    print("=" * 100)
    print(json.dumps(info, indent=2))
    try:
        table_text = summary[show].round(4).to_markdown(index=False)   # needs `tabulate`
    except ImportError:
        table_text = summary[show].round(4).to_string(index=False)
    with open(os.path.join(RESULTS_DIR, "metrics_summary.md"), "w") as f:
        f.write(table_text + "\n")


if __name__ == "__main__":
    main()
