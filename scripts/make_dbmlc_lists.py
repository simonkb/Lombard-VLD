import argparse
import os
import random
from collections import defaultdict


def _list_speakers(root):
    speakers = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        if len(name) >= 2 and name[0] in ('F', 'M') and name[1:].isdigit():
            speakers.append(name)
    return sorted(speakers)


def _parse_filename(name):
    # SPKR_Gxx_Uxx_SSNyy.wav
    base = os.path.splitext(name)[0]
    parts = base.split('_')
    if len(parts) != 4:
        return None
    spk, group, utt, ssn = parts
    if not (group.startswith('G') and utt.startswith('U') and ssn.startswith('SSN')):
        return None
    return spk, group, utt, ssn


def _build_pairs(root, plain_cond, lombard_conds):
    # returns: spk -> utt_id -> cond -> path
    pairs = defaultdict(lambda: defaultdict(dict))
    missing = 0
    speakers = _list_speakers(root)
    for spk in speakers:
        for cond in [plain_cond] + lombard_conds:
            cond_dir = os.path.join(root, spk, cond)
            if not os.path.isdir(cond_dir):
                continue
            for name in os.listdir(cond_dir):
                if not name.lower().endswith('.wav'):
                    continue
                parsed = _parse_filename(name)
                if not parsed:
                    continue
                pspk, group, utt, ssn = parsed
                if pspk != spk or ssn != cond:
                    continue
                utt_id = f"{group}_{utt}"
                pairs[spk][utt_id][cond] = os.path.join(cond_dir, name)
    return pairs, missing


def _speaker_split(speakers, seed, train_ratio):
    spk_list = sorted(speakers)
    rng = random.Random(seed)
    rng.shuffle(spk_list)
    n_train = int(round(len(spk_list) * train_ratio))
    train = spk_list[:n_train]
    test = spk_list[n_train:]
    return train, test


def _collect_pair_units(pairs_by_spk, plain_cond, lombard_conds):
    # returns list of (spk, utt_id, plain_path, lombard_path)
    units = []
    for spk, utts in pairs_by_spk.items():
        for utt_id, by_cond in utts.items():
            p = by_cond.get(plain_cond)
            if not p:
                continue
            for lc in lombard_conds:
                l = by_cond.get(lc)
                if not l:
                    continue
                units.append((spk, utt_id, p, l))
    return units


def _index_by_utt(units):
    by_utt = defaultdict(list)
    for spk, utt_id, p, l in units:
        by_utt[utt_id].append((spk, p, l))
    return by_utt


def _write_train_list(train_units, out_path, seed):
    rng = random.Random(seed)
    by_utt = _index_by_utt(train_units)

    positives = []
    for spk, utt_id, p, l in train_units:
        positives.append((1, p, l))

    negatives = []
    utts = list(by_utt.keys())
    if not utts:
        raise RuntimeError('No utterances with plain+Lombard pairs found for training.')

    max_tries = max(1000, len(positives) * 50)
    tries = 0
    while len(negatives) < len(positives) and tries < max_tries:
        tries += 1
        utt = rng.choice(utts)
        candidates = by_utt[utt]
        if len(candidates) < 2:
            continue
        (spk1, p1, _), (spk2, _, l2) = rng.sample(candidates, 2)
        if spk1 == spk2:
            continue
        negatives.append((0, p1, l2))

    if len(negatives) < len(positives):
        raise RuntimeError('Unable to generate enough negative training pairs.')

    with open(out_path, 'w', encoding='utf-8') as f:
        for label, ref, test in positives:
            f.write(f"{label}\t{ref}\t{test}\n")
        for label, ref, test in negatives:
            f.write(f"{label}\t{ref}\t{test}\n")

    return len(positives), len(negatives)


def _write_val_trials(val_units, out_path, seed):
    rng = random.Random(seed)
    by_spk = defaultdict(list)
    for spk, utt_id, p, l in val_units:
        by_spk[spk].append((utt_id, p, l))

    speaker_pairs = []
    for spk, items in by_spk.items():
        if len(items) >= 2:
            speaker_pairs.append((spk, items))

    positives = []
    for spk, items in speaker_pairs:
        for _ in range(min(10, len(items) // 2)):
            (utt1, p1, l1), (utt2, p2, l2) = rng.sample(items, 2)
            positives.append((1, p1, l1, p2, l2))
    if not positives:
        raise RuntimeError('Not enough validation pairs to form positive trials.')

    by_utt = _index_by_utt(val_units)
    negatives = []
    utts = list(by_utt.keys())
    if not utts:
        raise RuntimeError('No utterances with plain+Lombard pairs found for validation.')

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

    if len(negatives) < len(positives):
        # fallback: any-utterance different-speaker negatives
        flat = [(spk, p, l) for spk, utt_id, p, l in val_units]
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
        for label, r1, r2, t1, t2 in positives:
            f.write(f"{label} {r1} {r2} {t1} {t2}\n")
        for label, r1, r2, t1, t2 in negatives:
            f.write(f"{label} {r1} {r2} {t1} {t2}\n")

    return len(positives), len(negatives)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='DB-MLC root, containing speaker folders like F01, M23')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train_ratio', type=float, default=0.8)
    ap.add_argument('--plain_cond', type=str, default='SSN30')
    ap.add_argument('--lombard_conds', type=str, default='SSN80', help='Comma-separated, e.g. SSN55,SSN70,SSN80')
    args = ap.parse_args()

    lombard_conds = [c.strip() for c in args.lombard_conds.split(',') if c.strip()]
    if not lombard_conds:
        raise RuntimeError('No Lombard conditions provided.')

    os.makedirs(args.out_dir, exist_ok=True)

    pairs_by_spk, _ = _build_pairs(args.root, args.plain_cond, lombard_conds)
    all_speakers = sorted(pairs_by_spk.keys())
    if not all_speakers:
        raise RuntimeError('No speaker folders found under root.')

    train_spk, val_spk = _speaker_split(all_speakers, args.seed, args.train_ratio)

    train_pairs_by_spk = {s: pairs_by_spk[s] for s in train_spk}
    val_pairs_by_spk = {s: pairs_by_spk[s] for s in val_spk}

    train_units = _collect_pair_units(train_pairs_by_spk, args.plain_cond, lombard_conds)
    val_units = _collect_pair_units(val_pairs_by_spk, args.plain_cond, lombard_conds)

    train_list_path = os.path.join(args.out_dir, 'dbmlc_train_pairs.txt')
    val_trials_path = os.path.join(args.out_dir, 'dbmlc_val_trials.txt')

    npos_tr, nneg_tr = _write_train_list(train_units, train_list_path, args.seed)
    npos_val, nneg_val = _write_val_trials(val_units, val_trials_path, args.seed)

    print('Speakers:', len(all_speakers))
    print('Train speakers:', len(train_spk))
    print('Val speakers:', len(val_spk))
    print('Train pairs: pos', npos_tr, 'neg', nneg_tr)
    print('Val trials: pos', npos_val, 'neg', nneg_val)
    print('Train list:', train_list_path)
    print('Val trials:', val_trials_path)


if __name__ == '__main__':
    main()
