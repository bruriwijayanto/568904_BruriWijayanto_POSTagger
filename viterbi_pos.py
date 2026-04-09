"""
viterbi_pos.py
==============
Part-of-Speech Tagger untuk Bahasa Indonesia menggunakan
Hidden Markov Model (HMM) + Algoritma Viterbi.

Tugas 1 PBAL – Sequence Labeling
"""

import math
import random
import re
from collections import defaultdict


# ---------------------------------------------------------------------------
# 1. KONSTANTA
# ---------------------------------------------------------------------------

TAGS = [
    "NN", "NNP", "NND", "PRP", "PR",
    "SC", "MD", "FW", "SYM", "Z",
    "VB", "JJ", "RB", "IN", "CC",
    "CD", "OD", "NEG", "DT", "X",
]

K_SMOOTH = 0.001   # konstanta k-smoothing untuk emisi
TRAIN_RATIO = 0.8  # proporsi data training


# ---------------------------------------------------------------------------
# 2. PEMBACAAN DATA
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> list[list[tuple[str, str]]]:
    """
    Membaca file TSV berformat XML-sentence dan mengembalikan daftar kalimat.

    Setiap kalimat adalah list of (word, tag).
    Format file:
        <kalimat id=N>
        word\tTAG
        ...
        </kalimat>
    """
    sentences: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            stripped = line.strip()

            if stripped.startswith("<kalimat") and not stripped.startswith("</kalimat"):
                current = []
            elif stripped == "</kalimat>":
                if current:
                    sentences.append(current)
                    current = []
            elif stripped and "\t" in stripped:
                parts = stripped.split("\t")
                if len(parts) >= 2:
                    word = parts[0].strip()
                    tag  = parts[1].strip()
                    if word and tag:
                        current.append((word, tag))

    # Tangani kalimat tanpa tag penutup di akhir file
    if current:
        sentences.append(current)

    return sentences


def split_data(
    sentences: list, train_ratio: float = TRAIN_RATIO, seed: int = 42
) -> tuple[list, list]:
    """Membagi data secara acak menjadi train dan test set."""
    data = sentences[:]
    random.seed(seed)
    random.shuffle(data)
    split = int(len(data) * train_ratio)
    return data[:split], data[split:]


# ---------------------------------------------------------------------------
# 3. TRAINING HMM
# ---------------------------------------------------------------------------

def train_hmm(sentences: list[list[tuple[str, str]]]) -> tuple[dict, dict, dict]:
    """
    Mengestimasi parameter HMM dari data training.

    Parameter
    ---------
    sentences : list of list of (word, tag)

    Returns
    -------
    pi        : dict  tag -> log P(tag | awal kalimat)
    trans_prob: dict  (tag_i-1, tag_i) -> log P(tag_i | tag_i-1)  [Laplace]
    emit_prob : dict  (tag, word) -> log P(word | tag)             [k-smooth]
    """
    # ── hitung frekuensi ────────────────────────────────────────────────────
    init_count  : dict[str, int]               = defaultdict(int)
    trans_count : dict[tuple, int]             = defaultdict(int)
    tag_count   : dict[str, int]               = defaultdict(int)
    emit_count  : dict[tuple, int]             = defaultdict(int)
    vocab       : set[str]                     = set()

    for sent in sentences:
        if not sent:
            continue
        words, tags = zip(*sent)

        # initial
        init_count[tags[0]] += 1

        for i, (w, t) in enumerate(sent):
            tag_count[t]       += 1
            emit_count[(t, w)] += 1
            vocab.add(w)

            if i > 0:
                trans_count[(tags[i - 1], t)] += 1

    N = len(TAGS)
    V = len(vocab)

    # ── initial probability (log, dengan Laplace) ──────────────────────────
    total_init = sum(init_count.values())
    pi: dict[str, float] = {}
    for t in TAGS:
        pi[t] = math.log((init_count[t] + 1) / (total_init + N))

    # ── transition probability (log, Laplace / add-one) ────────────────────
    trans_prob: dict[tuple, float] = {}
    for t_prev in TAGS:
        denom = tag_count[t_prev] + N   # denominator dengan smoothing
        for t_cur in TAGS:
            numerator = trans_count[(t_prev, t_cur)] + 1
            trans_prob[(t_prev, t_cur)] = math.log(numerator / denom)

    # ── emission probability (log, k-smoothing) ────────────────────────────
    emit_prob: dict[tuple, float] = {}
    for t in TAGS:
        denom = tag_count[t] + K_SMOOTH * (V + 1)   # +1 untuk unknown
        for w in vocab:
            numerator = emit_count[(t, w)] + K_SMOOTH
            emit_prob[(t, w)] = math.log(numerator / denom)

    return pi, trans_prob, emit_prob


# ---------------------------------------------------------------------------
# 4. ALGORITMA VITERBI
# ---------------------------------------------------------------------------

def _emission_log_prob(
    tag: str,
    word: str,
    emit_prob: dict,
    vocab_size: int,
    tag_count: dict,
) -> float:
    """
    Mengembalikan log P(word | tag).
    Untuk unknown word digunakan uniform smoothing (k-smoothing dengan count 0).
    """
    key = (tag, word)
    if key in emit_prob:
        return emit_prob[key]

    # Unknown word: k / (count(tag) + k*(V+1))
    # Karena emit_prob sudah menyimpan nilai log untuk kata diketahui,
    # kita hitung ulang untuk unknown:
    count_t = tag_count.get(tag, 0)
    denom   = count_t + K_SMOOTH * (vocab_size + 1)
    return math.log(K_SMOOTH / denom) if denom > 0 else math.log(1e-300)


def viterbi(
    sentence : list[str],
    tags     : list[str],
    pi       : dict,
    trans_prob: dict,
    emit_prob : dict,
    tag_count : dict,
    vocab_size: int,
) -> list[str]:
    """
    Algoritma Viterbi dalam ruang log untuk menghindari numerical underflow.

    Parameter
    ---------
    sentence   : list of str  (kata-kata dalam kalimat)
    tags       : list of str  (daftar tag yang mungkin)
    pi         : log initial probabilities
    trans_prob : log transition probabilities
    emit_prob  : log emission probabilities
    tag_count  : raw count per tag (untuk unknown-word smoothing)
    vocab_size : ukuran vocabulary training

    Returns
    -------
    best_tags  : list of str  (urutan tag terbaik)
    """
    T = len(sentence)
    N = len(tags)
    NEG_INF = float("-inf")

    # δ[t][i] = log prob tertinggi untuk urutan dengan tag ke-i pada posisi t
    # ψ[t][i] = backpointer: index tag sebelumnya yang menghasilkan δ[t][i]
    delta: list[list[float]]   = [[NEG_INF] * N for _ in range(T)]
    psi  : list[list[int]]     = [[-1]       * N for _ in range(T)]

    # ── Inisialisasi (posisi pertama) ───────────────────────────────────────
    w0 = sentence[0]
    for i, t in enumerate(tags):
        e = _emission_log_prob(t, w0, emit_prob, vocab_size, tag_count)
        delta[0][i] = pi.get(t, math.log(1e-300)) + e
        psi[0][i]   = -1

    # ── Rekursi ─────────────────────────────────────────────────────────────
    for pos in range(1, T):
        w = sentence[pos]
        for cur_i, t_cur in enumerate(tags):
            e = _emission_log_prob(t_cur, w, emit_prob, vocab_size, tag_count)
            best_score = NEG_INF
            best_prev  = -1

            for prev_i, t_prev in enumerate(tags):
                score = (
                    delta[pos - 1][prev_i]
                    + trans_prob.get((t_prev, t_cur), math.log(1e-300))
                    + e
                )
                if score > best_score:
                    best_score = score
                    best_prev  = prev_i

            delta[pos][cur_i] = best_score
            psi[pos][cur_i]   = best_prev

    # ── Terminasi ────────────────────────────────────────────────────────────
    best_last  = max(range(N), key=lambda i: delta[T - 1][i])
    best_score = delta[T - 1][best_last]          # noqa: F841  (for reference)

    # ── Backtracking ─────────────────────────────────────────────────────────
    best_seq = [best_last]
    for pos in range(T - 1, 0, -1):
        best_seq.append(psi[pos][best_seq[-1]])
    best_seq.reverse()

    return [tags[i] for i in best_seq]


# ---------------------------------------------------------------------------
# 5. EVALUASI
# ---------------------------------------------------------------------------

def evaluate(
    test_sentences: list[list[tuple[str, str]]],
    tags          : list[str],
    pi            : dict,
    trans_prob    : dict,
    emit_prob     : dict,
    tag_count     : dict,
    vocab_size    : int,
    verbose       : bool = False,
) -> float:
    """
    Menghitung token-level accuracy pada test set.

    Returns
    -------
    accuracy : float  (0–1)
    """
    total_tokens   = 0
    correct_tokens = 0

    # Untuk analisis kesalahan
    confusion: dict[tuple, int] = defaultdict(int)

    for sent in test_sentences:
        if not sent:
            continue
        words    = [w for w, _ in sent]
        true_tags= [t for _, t in sent]
        pred_tags = viterbi(words, tags, pi, trans_prob, emit_prob, tag_count, vocab_size)

        for true_t, pred_t in zip(true_tags, pred_tags):
            total_tokens += 1
            if true_t == pred_t:
                correct_tokens += 1
            else:
                confusion[(true_t, pred_t)] += 1

    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

    if verbose:
        print(f"\n{'='*60}")
        print(f"  HASIL EVALUASI")
        print(f"{'='*60}")
        print(f"  Total token  : {total_tokens:,}")
        print(f"  Benar        : {correct_tokens:,}")
        print(f"  Akurasi      : {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\n  Top-10 konfusi (true → pred : count):")
        top = sorted(confusion.items(), key=lambda x: -x[1])[:10]
        for (true_t, pred_t), cnt in top:
            print(f"    {true_t:>5} → {pred_t:<5} : {cnt}")
        print(f"{'='*60}\n")

    return accuracy


# ---------------------------------------------------------------------------
# 6. PREDIKSI KALIMAT BARU
# ---------------------------------------------------------------------------

def predict_sentence(
    raw_sentence: str,
    tags        : list[str],
    pi          : dict,
    trans_prob  : dict,
    emit_prob   : dict,
    tag_count   : dict,
    vocab_size  : int,
) -> list[tuple[str, str]]:
    """
    Memprediksi tag POS untuk kalimat mentah (string).

    Parameter
    ---------
    raw_sentence : str  – kalimat input, dipisah spasi

    Returns
    -------
    list of (word, predicted_tag)
    """
    words     = raw_sentence.strip().split()
    pred_tags = viterbi(words, tags, pi, trans_prob, emit_prob, tag_count, vocab_size)
    return list(zip(words, pred_tags))


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

def main():
    import os

    # ── Lokasi file data ─────────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file  = os.path.join(script_dir, "Data_POS-Tag-ID.tsv")

    print("=" * 60)
    print("  HMM POS Tagger – Bahasa Indonesia")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\n[1] Membaca data dari: {data_file}")
    sentences = load_data(data_file)
    print(f"    Total kalimat : {len(sentences):,}")
    total_tokens = sum(len(s) for s in sentences)
    print(f"    Total token   : {total_tokens:,}")

    # ── Split data ────────────────────────────────────────────────────────────
    print(f"\n[2] Membagi data ({int(TRAIN_RATIO*100)}/{int((1-TRAIN_RATIO)*100)}) ...")
    train_sents, test_sents = split_data(sentences)
    print(f"    Kalimat training : {len(train_sents):,}")
    print(f"    Kalimat testing  : {len(test_sents):,}")

    # ── Training ──────────────────────────────────────────────────────────────
    print("\n[3] Training HMM (estimasi parameter) ...")
    pi, trans_prob, emit_prob = train_hmm(train_sents)

    # Hitung tag_count dan vocab_size untuk inferensi
    tag_count: dict[str, int] = defaultdict(int)
    vocab: set[str]           = set()
    for sent in train_sents:
        for w, t in sent:
            tag_count[t] += 1
            vocab.add(w)
    vocab_size = len(vocab)

    print(f"    Ukuran vocab     : {vocab_size:,}")
    print(f"    Jumlah tag       : {len(TAGS)}")

    # ── Evaluasi ──────────────────────────────────────────────────────────────
    print("\n[4] Evaluasi pada test set ...")
    accuracy = evaluate(
        test_sents, TAGS, pi, trans_prob, emit_prob,
        tag_count, vocab_size, verbose=True
    )

    # ── Contoh prediksi ───────────────────────────────────────────────────────
    print("[5] Contoh prediksi kalimat:")
    examples = [
        "Pemerintah kota Jakarta membangun jalan tol .",
        "Saya tidak bisa pergi ke sekolah hari ini .",
        "Bill Gates dan Warren Buffett sangat kaya .",
    ]
    for ex in examples:
        result = predict_sentence(
            ex, TAGS, pi, trans_prob, emit_prob, tag_count, vocab_size
        )
        print(f"\n  Input : {ex}")
        print(f"  Output: {' '.join(f'{w}/{t}' for w, t in result)}")

    print("\nSelesai.")


if __name__ == "__main__":
    main()
