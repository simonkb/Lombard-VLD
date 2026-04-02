import argparse
import os
import random
import re


_WAV_RE = re.compile(r"^(?P<spk>[FM]\d+)_U(?P<utt>\d{3})_(?P<ssn>SSN\d{2})_")


def _iter_wavs(roots):
	for root in roots:
		for dirpath, _, filenames in os.walk(root):
			for fn in filenames:
				if fn.lower().endswith('.wav'):
					yield os.path.join(dirpath, fn)


def _parse_info(path: str):
	m = _WAV_RE.match(os.path.basename(path))
	if not m:
		return None
	spk = m.group('spk')
	utt = m.group('utt')
	ssn = m.group('ssn')
	# Only keep SSN40 (plain) and SSN80 (lombard). Ignore SSN55 for now.
	if ssn not in ('SSN40', 'SSN80'):
		return None
	return spk, utt, ssn


def _index_by_speaker_utt_ssn(roots):
	by_spk = {}
	for wav in _iter_wavs(roots):
		info = _parse_info(wav)
		if info is None:
			continue
		spk, utt, ssn = info
		by_spk.setdefault(spk, {}).setdefault(utt, {}).setdefault(ssn, []).append(wav)
	return by_spk


def _sample_two_distinct(rng: random.Random, wavs):
	return rng.sample(wavs, 2)


def _sample_one(rng: random.Random, wavs):
	return rng.choice(wavs)


def _collect_utt_keys(by_spk):
	keys = []
	for spk, by_utt in by_spk.items():
		for utt, by_ssn in by_utt.items():
			if by_ssn.get('SSN40') and by_ssn.get('SSN80'):
				keys.append((spk, utt))
	return keys


def _write_trials(
	rng: random.Random,
	by_spk,
	keys,
	out_path: str,
	num_trials: int,
) -> None:
	if len(keys) < 2:
		raise ValueError('Need at least 2 utterances with both SSN40 and SSN80 to form trials')

	def _pick_other_key(spk_a, utt_a):
		# Prefer a different speaker if possible; otherwise just ensure different (spk, utt).
		keys_other_spk = [(s, u) for (s, u) in keys if s != spk_a]
		pool = keys_other_spk if keys_other_spk else keys
		spk_b, utt_b = rng.choice(pool)
		while (spk_b, utt_b) == (spk_a, utt_a):
			spk_b, utt_b = rng.choice(pool)
		return spk_b, utt_b

	def _pair_40_80(spk, utt):
		plain = _sample_one(rng, by_spk[spk][utt]['SSN40'])
		lomb = _sample_one(rng, by_spk[spk][utt]['SSN80'])
		return plain, lomb

	with open(out_path, 'w') as fo:
		for i in range(num_trials):
			label = 1 if (i % 2 == 0) else 0
			spk_a, utt_a = rng.choice(keys)
			a1, a2 = _pair_40_80(spk_a, utt_a)
			if label == 1:
				spk_b, utt_b = rng.choice(keys)
				while (spk_b, utt_b) == (spk_a, utt_a):
					spk_b, utt_b = rng.choice(keys)
				b1, b2 = _pair_40_80(spk_b, utt_b)
				fo.write(f'1\t{a1}\t{a2}\t{b1}\t{b2}\n')
			else:
				spk_b, utt_b = _pick_other_key(spk_a, utt_a)
				b_plain = _sample_one(rng, by_spk[spk_b][utt_b]['SSN40'])
				fo.write(f'0\t{a1}\t{a2}\t{b_plain}\t{b_plain}\n')


def generate_emalg_val_pair_list(
	roots,
	out_path: str,
	num_trials: int = 5000,
	seed: int = 0,
	within_speaker: bool = True,
) -> None:
	"""Generate a 5-column trial list compatible with ECAPAModel.eval_network.

	Protocol (mirrors the structure of val_apple_pair_list_replay_1_2.txt):
	- label 1: (SSN40, SSN80) vs (SSN40, SSN80)
	- label 0: (SSN40, SSN80) vs (SSN40, SSN40)

	SSN55 is ignored.
	"""
	rng = random.Random(seed)
	by_spk = _index_by_speaker_utt_ssn(roots)
	if not by_spk:
		raise ValueError(f'No wavs found under roots={roots}')

	speakers = sorted(by_spk.keys())

	def _spk_has_40_80_pair(spk: str) -> bool:
		for utt, by_ssn in by_spk.get(spk, {}).items():
			if by_ssn.get('SSN40') and by_ssn.get('SSN80'):
				return True
		return False

	def _spk_has_40_40_pair(spk: str) -> bool:
		for _, by_ssn in by_spk.get(spk, {}).items():
			if len(by_ssn.get('SSN40', [])) >= 2:
				return True
		return False

	def _pick_spk_for(require_40_80: bool, require_40_40: bool) -> str:
		cands = []
		for s in speakers:
			if require_40_80 and (not _spk_has_40_80_pair(s)):
				continue
			if require_40_40 and (not _spk_has_40_40_pair(s)):
				continue
			cands.append(s)
		if not cands:
			raise ValueError('Not enough EMALG wavs to satisfy requested pairing constraints')
		return rng.choice(cands)

	def _sample_40_80_pair(spk: str):
		utts = [u for u, by_ssn in by_spk[spk].items() if by_ssn.get('SSN40') and by_ssn.get('SSN80')]
		utt = rng.choice(utts)
		plain = _sample_one(rng, by_spk[spk][utt]['SSN40'])
		lomb = _sample_one(rng, by_spk[spk][utt]['SSN80'])
		return plain, lomb

	def _sample_40_40_pair_same_text(spk: str):
		utts = [u for u, by_ssn in by_spk[spk].items() if len(by_ssn.get('SSN40', [])) >= 2]
		if not utts:
			raise ValueError(
				f'Cannot form (SSN40,SSN40) with identical text for speaker={spk}. '
				'No utterance has >=2 SSN40 recordings.'
			)
		utt = rng.choice(utts)
		a, b = _sample_two_distinct(rng, by_spk[spk][utt]['SSN40'])
		return a, b

	# Kept for backward compatibility. Prefer the train/val list generator below.
	keys = _collect_utt_keys(by_spk)
	_write_trials(rng=rng, by_spk=by_spk, keys=keys, out_path=out_path, num_trials=num_trials)


def generate_emalg_train_val_lists(
	roots,
	train_out: str = 'emalg_train_file_list.txt',
	val_out: str = 'emalg_val_file_list.txt',
	seed: int = 0,
	val_ratio: float = 0.2,
	trials_per_utt: int = 1,
) -> None:
	rng = random.Random(seed)
	by_spk = _index_by_speaker_utt_ssn(roots)
	keys = _collect_utt_keys(by_spk)
	if not keys:
		raise ValueError(f'No usable utterances found under roots={roots}')
	keys = sorted(keys)
	rng.shuffle(keys)

	n_val = int(round(len(keys) * val_ratio))
	n_val = max(1, min(len(keys) - 1, n_val))
	val_keys = keys[:n_val]
	train_keys = keys[n_val:]

	_write_trials(
		rng=rng,
		by_spk=by_spk,
		keys=train_keys,
		out_path=train_out,
		num_trials=len(train_keys) * max(1, trials_per_utt),
	)
	_write_trials(
		rng=rng,
		by_spk=by_spk,
		keys=val_keys,
		out_path=val_out,
		num_trials=len(val_keys) * max(1, trials_per_utt),
	)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate EMALG 5-column validation trial list')
	parser.add_argument(
		'--roots',
		nargs='+',
		type=str,
		default=['enhanced_lombard_grid_part1', 'enhanced_lombard_grid_part2'],
	)
	parser.add_argument('--train_out', type=str, default='emalg_train_file_list.txt')
	parser.add_argument('--val_out', type=str, default='emalg_val_file_list.txt')
	parser.add_argument('--val_ratio', type=float, default=0.2)
	parser.add_argument('--trials_per_utt', type=int, default=1)
	parser.add_argument('--seed', type=int, default=0)

	args = parser.parse_args()
	generate_emalg_train_val_lists(
		roots=args.roots,
		train_out=args.train_out,
		val_out=args.val_out,
		seed=args.seed,
		val_ratio=args.val_ratio,
		trials_per_utt=args.trials_per_utt,
	)
	print(args.train_out)
	print(args.val_out)
