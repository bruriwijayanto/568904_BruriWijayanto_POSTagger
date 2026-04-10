from collections import Counter, defaultdict

from viterbi_pos import TAGS, load_data, split_data, train_hmm, viterbi


def main() -> None:
    sentences = load_data("Data_POS-Tag-ID.tsv")
    train_sents, test_sents = split_data(sentences)

    train_vocab = {word for sent in train_sents for word, _ in sent}
    tag_count = Counter(tag for sent in train_sents for _, tag in sent)

    pi, trans_prob, emit_prob = train_hmm(train_sents)

    total_tokens = 0
    correct_tokens = 0
    known_total = 0
    known_correct = 0
    unknown_total = 0
    unknown_correct = 0

    confusion = Counter()
    per_tag = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    error_sentences = []

    for sent in test_sents:
        words = [word for word, _ in sent]
        gold_tags = [tag for _, tag in sent]
        pred_tags = viterbi(
            words, TAGS, pi, trans_prob, emit_prob, tag_count, len(train_vocab)
        )

        errors = []
        for word, gold, pred in zip(words, gold_tags, pred_tags):
            total_tokens += 1
            is_unknown = word not in train_vocab

            if gold == pred:
                correct_tokens += 1
                per_tag[gold]["tp"] += 1
                if is_unknown:
                    unknown_correct += 1
                else:
                    known_correct += 1
            else:
                confusion[(gold, pred)] += 1
                per_tag[gold]["fn"] += 1
                per_tag[pred]["fp"] += 1
                errors.append((word, gold, pred, is_unknown))

            if is_unknown:
                unknown_total += 1
            else:
                known_total += 1

        if errors:
            error_sentences.append((len(errors), words, gold_tags, pred_tags, errors))

    print("DATASET")
    print(f"total_sentences={len(sentences)}")
    print(f"train_sentences={len(train_sents)}")
    print(f"test_sentences={len(test_sents)}")
    print(f"total_tokens={sum(len(s) for s in sentences)}")
    print(f"train_tokens={sum(len(s) for s in train_sents)}")
    print(f"test_tokens={sum(len(s) for s in test_sents)}")
    print(f"train_vocab={len(train_vocab)}")
    print(f"unique_tags_in_data={sorted({tag for sent in sentences for _, tag in sent})}")
    print()

    print("EVALUATION")
    print(f"token_accuracy={correct_tokens / total_tokens:.6f}")
    print(f"correct_tokens={correct_tokens}")
    print(f"unknown_total={unknown_total}")
    print(f"unknown_accuracy={unknown_correct / unknown_total:.6f}")
    print(f"known_total={known_total}")
    print(f"known_accuracy={known_correct / known_total:.6f}")
    print()

    print("TOP_CONFUSIONS")
    for (gold, pred), count in confusion.most_common(15):
        print(f"{gold}->{pred}={count}")
    print()

    print("PER_TAG_F1")
    rows = []
    for tag in sorted(per_tag):
        tp = per_tag[tag]["tp"]
        fp = per_tag[tag]["fp"]
        fn = per_tag[tag]["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append((f1, tag, precision, recall, tp + fn))
    for f1, tag, precision, recall, support in sorted(rows, reverse=True):
        print(
            f"{tag}: precision={precision:.4f}, recall={recall:.4f}, "
            f"f1={f1:.4f}, support={support}"
        )
    print()

    print("ERROR_EXAMPLES")
    error_sentences.sort(key=lambda item: (-item[0], " ".join(item[1])))
    for error_count, words, gold_tags, pred_tags, errors in error_sentences[:20]:
        print(f"sentence={' '.join(words)}")
        print(f"error_count={error_count}")
        print(
            "gold="
            + " ".join(f"{word}/{tag}" for word, tag in zip(words, gold_tags))
        )
        print(
            "pred="
            + " ".join(f"{word}/{tag}" for word, tag in zip(words, pred_tags))
        )
        print("errors=" + "; ".join(
            f"{word}:{gold}->{pred}:unknown={is_unknown}"
            for word, gold, pred, is_unknown in errors
        ))
        print()


if __name__ == "__main__":
    main()
