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

    # Spoof negatives: plain + plain (same speaker + same utterance)
    negatives = []
    for spk, utt, p, l in train_units:
        negatives.append((0, p, p))

    with open(out_path, 'w', encoding='utf-8') as f:
        for label, ref, test in positives:
            f.write(f"{label}\t{ref}\t{test}\n")
        for label, ref, test in negatives:
            f.write(f"{label}\t{ref}\t{test}\n")

    return len(positives), len(negatives)


def _write_val_trials(val_units, out_path, seed):
    rng = random.Random(seed)
    # Live pairs (plain+lombard) from all valid units
    live_pairs = [(p, l) for spk, utt, p, l in val_units]
    if len(live_pairs) < 2:
        raise RuntimeError('Not enough validation pairs to form trials.')

    # Spoof pairs: plain + plain (same speaker + utterance)
    spoof_pairs = []
    for spk, utt, p, l in val_units:
        spoof_pairs.append((p, p))

    positives = []
    negatives = []
    max_pairs = min(len(live_pairs), len(spoof_pairs))
    if max_pairs == 0:
        raise RuntimeError('No live/spoof pairs available for validation.')

    # Sample trials: live-live (label 1) and live-spoof (label 0)
    n_trials = min(110, max_pairs // 2)
    for _ in range(n_trials):
        (p1, l1), (p2, l2) = rng.sample(live_pairs, 2)
        positives.append((1, p1, l1, p2, l2))
        (p3, l3) = rng.choice(live_pairs)
        (p4, l4) = rng.choice(spoof_pairs)
        negatives.append((0, p3, l3, p4, l4))

    with open(out_path, 'w', encoding='utf-8') as f:
        # write in 5-column format: label ref1 ref2 test1 test2
        for label, r1, r2, t1, t2 in positives:
            f.write(f"{label} {r1} {r2} {t1} {t2}\n")
        for label, r1, r2, t1, t2 in negatives:
            f.write(f"{label} {r1} {r2} {t1} {t2}\n")

    return len(positives), len(negatives)


def _sanity_report(train_path, val_path, sample_n=5):
    def _parse_name(p):
        base = os.path.basename(p)
        # s2_p_abcdef.wav
        parts = base.split('_')
        if len(parts) < 3:
            return '', '', ''
        spk = parts[0]
        cond = parts[1]
        utt = os.path.splitext(parts[2])[0]
        return spk, cond, utt

    live_train = 0
    spoof_train = 0
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 3:
                continue
            label = parts[0]
            if label == '1':
                live_train += 1
            elif label == '0':
                spoof_train += 1

    live_trials = 0
    spoof_trials = 0
    with open(val_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split(' ')
            if len(parts) != 5:
                continue
            label = parts[0]
            if label == '1':
                live_trials += 1
            elif label == '0':
                spoof_trials += 1

    print('Sanity report')
    print('Train pairs: live', live_train, 'spoof', spoof_train)
    print('Val trials: live-live', live_trials, 'live-spoof', spoof_trials)

    print('Train samples')
    with open(train_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_n:
                break
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 3:
                continue
            label, a, b = parts
            s1, c1, u1 = _parse_name(a)
            s2, c2, u2 = _parse_name(b)
            print(f"{label}\t{s1}:{c1}:{u1}\t{s2}:{c2}:{u2}")

    print('Val samples')
    with open(val_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_n:
                break
            parts = line.rstrip('\n').split(' ')
            if len(parts) != 5:
                continue
            label, a, b, c, d = parts
            s1, c1, u1 = _parse_name(a)
            s2, c2, u2 = _parse_name(b)
            s3, c3, u3 = _parse_name(c)
            s4, c4, u4 = _parse_name(d)
            print(f"{label}\t{s1}:{c1}:{u1}\t{s2}:{c2}:{u2}\t{s3}:{c3}:{u3}\t{s4}:{c4}:{u4}")


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
    _sanity_report(train_list_path, val_trials_path)


if __name__ == '__main__':
    main()
