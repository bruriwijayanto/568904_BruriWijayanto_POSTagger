# POS Tagger Bahasa Indonesia – HMM + Viterbi

Implementasi **Part-of-Speech (POS) Tagger** untuk Bahasa Indonesia menggunakan **Hidden Markov Model (HMM)** dan **Algoritma Viterbi**.

---

## Persyaratan

- Python **3.10+** (menggunakan type hint `list[...]` bawaan)
- Tidak membutuhkan library tambahan (murni Python standard library)

---

## Cara Menjalankan

### 1. Pastikan file data tersedia

Letakkan file `Data_POS-Tag-ID.tsv` pada direktori yang sama dengan `viterbi_pos.py`.

### 2. Jalankan program

```bash
python3 viterbi_pos.py
```

### 3. Output yang diharapkan

Program akan menampilkan:

```
============================================================
  HMM POS Tagger – Bahasa Indonesia
============================================================

[1] Membaca data dari: ...
    Total kalimat : 10,030
    Total token   : 256,622

[2] Membagi data (80/20) ...
    Kalimat training : 8,024
    Kalimat testing  : 2,006

[3] Training HMM (estimasi parameter) ...
    Ukuran vocab     : 16,462
    Jumlah tag       : 20

[4] Evaluasi pada test set ...

============================================================
  HASIL EVALUASI
============================================================
  Total token  : 51,206
  Benar        : 48,345
  Akurasi      : 0.9441 (94.41%)
  ...
============================================================

[5] Contoh prediksi kalimat:
  ...
```

---

## Struktur Kode

```
viterbi_pos.py
├── load_data()         – Parsing file TSV berformat XML-sentence
├── split_data()        – Pembagian data train/test (80/20)
├── train_hmm()         – Estimasi π, transisi (Laplace), emisi (k-smooth)
├── viterbi()           – Algoritma Viterbi dalam ruang log
├── evaluate()          – Token-level accuracy + analisis konfusi
├── predict_sentence()  – Prediksi POS tag dari kalimat mentah
└── main()              – Pipeline lengkap
```

---

## Daftar Tag POS

| Tag  | Keterangan              |
|------|-------------------------|
| NN   | Common Noun             |
| NNP  | Proper Noun             |
| NND  | Classifier/Measure      |
| PRP  | Pronoun                 |
| PR   | Demonstrative Pronoun   |
| SC   | Subordinating Conj.     |
| MD   | Modal                   |
| FW   | Foreign Word            |
| SYM  | Symbol                  |
| Z    | Punctuation             |
| VB   | Verb                    |
| JJ   | Adjective               |
| RB   | Adverb                  |
| IN   | Preposition             |
| CC   | Coordinating Conj.      |
| CD   | Cardinal Number         |
| OD   | Ordinal Number          |
| NEG  | Negation                |
| DT   | Determiner              |
| X    | Others                  |

---

## Parameter HMM

| Parameter         | Metode                                      |
|-------------------|---------------------------------------------|
| Initial prob. (π) | Laplace smoothing                           |
| Transisi          | Add-one (Laplace) smoothing                 |
| Emisi             | k-smoothing (k=0.001)                       |
| Unknown words     | k-smoothing dengan count=0 (uniform-like)   |

---

## Hasil Evaluasi

| Metrik          | Nilai       |
|-----------------|-------------|
| Token akurasi   | **94.41%**  |
| Total token uji | 51,206      |
| Token benar     | 48,345      |
