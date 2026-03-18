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

	with open(out_path, 'w') as fo:
		for _ in range(num_trials):
			is_match = rng.random() < 0.5
			if is_match:
				# Enforce identical spoken text by duplicating the exact same files.
				spk = _pick_spk_for(require_40_80=True, require_40_40=False)
				a1, a2 = _sample_40_80_pair(spk)
				fo.write(f'1\t{a1}\t{a2}\t{a1}\t{a2}\n')
			else:
				# Enforce identical spoken text by duplicating the SSN40 file.
				spk = _pick_spk_for(require_40_80=True, require_40_40=False)
				a1, a2 = _sample_40_80_pair(spk)
				fo.write(f'0\t{a1}\t{a2}\t{a1}\t{a1}\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Generate EMALG 5-column validation trial list')
	parser.add_argument(
		'--roots',
		nargs='+',
		type=str,
		default=['enhanced_lombard_grid_part1', 'enhanced_lombard_grid_part2'],
	)
	parser.add_argument(
		'--out',
		type=str,
		default='emalg_val_pair_list_plainplain_vs_lombardlombard.txt',
	)
	parser.add_argument('--num_trials', type=int, default=5000)
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument(
		'--within_speaker',
		action='store_true',
		help='If set, sample both pairs from the same speaker to reduce speaker leakage.',
	)

	args = parser.parse_args()
	generate_emalg_val_pair_list(
		roots=args.roots,
		out_path=args.out,
		num_trials=args.num_trials,
		seed=args.seed,
		within_speaker=args.within_speaker,
	)
	print(args.out)
