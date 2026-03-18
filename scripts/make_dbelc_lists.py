import argparse
import json
import os
import random
from collections import defaultdict


def _load_metadata(json_root):
    speakers = defaultdict(list)
    for name in os.listdir(json_root):
        if not name.lower().endswith('.json'):
            continue
        path = os.path.join(json_root, name)
        with open(path, 'r', encoding='utf-8') as f:
            try:
                items = json.load(f)
            except Exception as e:
                raise RuntimeError(f"Failed to parse {path}: {e}")
        for row in items:
            speakers[row.get('SPKR')].append(row)
    return speakers


def _build_pairs(speakers_rows, audio_root):
    # returns: spk -> utt -> {'p': path, 'l': path}
    pairs = defaultdict(lambda: defaultdict(dict))
    missing = 0
    for spk, rows in speakers_rows.items():
        for row in rows:
            if row.get('STATUS') != 'CORRECT':
                continue
            cond = row.get('COND')
            utt = row.get('UTTERANCE')
            if cond not in ('p', 'l'):
                continue
            wav = f"{spk}_{cond}_{utt}.wav"
            path = os.path.join(audio_root, wav)
            if not os.path.isfile(path):
                missing += 1
                continue
            pairs[spk][utt][cond] = path
    return pairs, missing


def _speaker_split(speakers, seed, train_ratio):
    spk_list = sorted(speakers)
    rng = random.Random(seed)
    rng.shuffle(spk_list)
    n_train = int(round(len(spk_list) * train_ratio))
    train = spk_list[:n_train]
    test = spk_list[n_train:]
    return train, test


def _collect_pair_units(pairs_by_spk):
    # returns list of (spk, utt, p_path, l_path)
    units = []
    for spk, utts in pairs_by_spk.items():
        for utt, by_cond in utts.items():
            p = by_cond.get('p')
            l = by_cond.get('l')
            if p and l:
                units.append((spk, utt, p, l))
    return units


def _index_by_utt(units):
    by_utt = defaultdict(list)
    for spk, utt, p, l in units:
        by_utt[utt].append((spk, p, l))
    return by_utt


def _write_train_list(train_units, out_path, seed):
    rng = random.Random(seed)
    by_utt = _index_by_utt(train_units)

    positives = []
    for spk, utt, p, l in train_units:
        positives.append((1, p, l))

    negatives = []
    # sample negatives matching positives count
    utts = list(by_utt.keys())
    if not utts:
        raise RuntimeError('No utterances with both p and l found for training.')

    while len(negatives) < len(positives):
        utt = rng.choice(utts)
        candidates = by_utt[utt]
        if len(candidates) < 2:
            continue
        (spk1, p1, _), (spk2, _, l2) = rng.sample(candidates, 2)
        if spk1 == spk2:
            continue
        negatives.append((0, p1, l2))

    with open(out_path, 'w', encoding='utf-8') as f:
        for label, ref, test in positives:
            f.write(f"{label}\t{ref}\t{test}\n")
        for label, ref, test in negatives:
            f.write(f"{label}\t{ref}\t{test}\n")

    return len(positives), len(negatives)


def _write_val_trials(val_units, out_path, seed):
    rng = random.Random(seed)
    by_spk = defaultdict(list)
    for spk, utt, p, l in val_units:
        by_spk[spk].append((utt, p, l))

    # build pair-units per speaker
    speaker_pairs = []
    for spk, items in by_spk.items():
        if len(items) >= 2:
            speaker_pairs.append((spk, items))

    positives = []
    for spk, items in speaker_pairs:
        # sample pairs within speaker
        for _ in range(min(10, len(items) // 2)):
            (utt1, p1, l1), (utt2, p2, l2) = rng.sample(items, 2)
            positives.append((1, p1, l1, p2, l2))
    if not positives:
        raise RuntimeError('Not enough validation pairs to form positive trials.')

    # negatives: different speakers, same utt if possible
    by_utt = _index_by_utt(val_units)
    negatives = []
    utts = list(by_utt.keys())
    if not utts:
        raise RuntimeError('No utterances with both p and l found for validation.')

    # First try: same-utterance, different-speaker negatives
    max_tries = max(1000, len(positives) * 50)
    tries = 0
    while len(negatives) < len(positives) and tries < max_tries:
        tries += 1
        utt = rng.choice(utts)
        candidates = by_utt[utt]
        if len(candidates) < 2:
            continue
        (spk1, p1, l1), (spk2, p2, l2) = rng.sample(candidates, 2)
        if spk1 == spk2:
            continue
        negatives.append((0, p1, l1, p2, l2))

    # Fallback: different speakers, any utterance
    if len(negatives) < len(positives):
        flat = [(spk, p, l) for spk, utt, p, l in val_units]
        if len(flat) < 2:
            raise RuntimeError('Not enough validation pairs to form negatives.')
        max_tries = max(1000, len(positives) * 50)
        tries = 0
        while len(negatives) < len(positives) and tries < max_tries:
            tries += 1
            (spk1, p1, l1), (spk2, p2, l2) = rng.sample(flat, 2)
            if spk1 == spk2:
                continue
            negatives.append((0, p1, l1, p2, l2))

    if len(negatives) < len(positives):
        raise RuntimeError(
            f'Unable to generate enough negatives: got {len(negatives)} needed {len(positives)}.'
        )

    with open(out_path, 'w', encoding='utf-8') as f:
        # write in 5-column format: label ref1 ref2 test1 test2
        for label, r1, r2, t1, t2 in positives:
            f.write(f"{label} {r1} {r2} {t1} {t2}\n")
        for label, r1, r2, t1, t2 in negatives:
            f.write(f"{label} {r1} {r2} {t1} {t2}\n")

    return len(positives), len(negatives)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json_root', required=True)
    ap.add_argument('--audio_root', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train_ratio', type=float, default=0.8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    speakers_rows = _load_metadata(args.json_root)
    pairs_by_spk, missing = _build_pairs(speakers_rows, args.audio_root)

    all_speakers = sorted(pairs_by_spk.keys())
    train_spk, val_spk = _speaker_split(all_speakers, args.seed, args.train_ratio)

    train_pairs_by_spk = {s: pairs_by_spk[s] for s in train_spk}
    val_pairs_by_spk = {s: pairs_by_spk[s] for s in val_spk}

    train_units = _collect_pair_units(train_pairs_by_spk)
    val_units = _collect_pair_units(val_pairs_by_spk)

    train_list_path = os.path.join(args.out_dir, 'dbelc_train_pairs.txt')
    val_trials_path = os.path.join(args.out_dir, 'dbelc_val_trials.txt')

    npos_tr, nneg_tr = _write_train_list(train_units, train_list_path, args.seed)
    npos_val, nneg_val = _write_val_trials(val_units, val_trials_path, args.seed)

    print('Speakers:', len(all_speakers))
    print('Train speakers:', len(train_spk))
    print('Val speakers:', len(val_spk))
    print('Missing audio files skipped:', missing)
    print('Train pairs: pos', npos_tr, 'neg', nneg_tr)
    print('Val trials: pos', npos_val, 'neg', nneg_val)
    print('Train list:', train_list_path)
    print('Val trials:', val_trials_path)


if __name__ == '__main__':
    main()
