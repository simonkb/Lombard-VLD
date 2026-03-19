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

    # Spoof negatives: plain + plain (same speaker + same utterance)
    negatives = []
    for spk, utt_id, p, l in train_units:
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
    live_pairs = [(p, l) for spk, utt_id, p, l in val_units]
    if len(live_pairs) < 2:
        raise RuntimeError('Not enough validation pairs to form trials.')

    # Spoof pairs: plain + plain (same speaker + utterance)
    spoof_pairs = []
    for spk, utt_id, p, l in val_units:
        spoof_pairs.append((p, p))

    positives = []
    negatives = []
    max_pairs = min(len(live_pairs), len(spoof_pairs))
    if max_pairs == 0:
        raise RuntimeError('No live/spoof pairs available for validation.')

    n_trials = min(110, max_pairs // 2)
    for _ in range(n_trials):
        (p1, l1), (p2, l2) = rng.sample(live_pairs, 2)
        positives.append((1, p1, l1, p2, l2))
        (p3, l3) = rng.choice(live_pairs)
        (p4, l4) = rng.choice(spoof_pairs)
        negatives.append((0, p3, l3, p4, l4))

    with open(out_path, 'w', encoding='utf-8') as f:
        for label, r1, r2, t1, t2 in positives:
            f.write(f"{label} {r1} {r2} {t1} {t2}\n")
        for label, r1, r2, t1, t2 in negatives:
            f.write(f"{label} {r1} {r2} {t1} {t2}\n")

    return len(positives), len(negatives)


def _sanity_report(train_path, val_path, sample_n=5):
    def _parse_name(p):
        base = os.path.basename(p)
        # SPKR_Gxx_Uxx_SSNyy.wav
        stem = os.path.splitext(base)[0]
        parts = stem.split('_')
        if len(parts) != 4:
            return '', '', ''
        spk, group, utt, ssn = parts
        return spk, ssn, f"{group}_{utt}"

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
    _sanity_report(train_list_path, val_trials_path)


if __name__ == '__main__':
    main()
