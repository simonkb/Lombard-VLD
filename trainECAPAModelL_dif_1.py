import argparse, os, time, warnings
import re
import random

import torch

from dataLoaderL_dif_wav import train_loader
from ECAPAModelL_dif_1_wav import ECAPAModel
from tools import init_args


def _resolve_device(device_str: str) -> str:
	if isinstance(device_str, str):
		if device_str.startswith('mps'):
			if torch.backends.mps.is_available():
				return 'mps'
			return 'cpu'
		if device_str.startswith('cuda'):
			if torch.cuda.is_available():
				return device_str
			return 'cpu'
	return str(device_str)


def _generate_train_list_from_pair_list(pair_list_path: str, out_path: str) -> None:
	with open(pair_list_path, 'r') as f:
		lines = f.readlines()

	with open(out_path, 'w') as fo:
		for line in lines:
			parts = line.strip().split('\t')
			if len(parts) != 5:
				raise ValueError(f"Expected 5 tab-separated columns in pair list, got {len(parts)}: {line[:200]}")
			label, ref1, ref2, test1, test2 = parts
			fo.write(f"{label}\t{ref1}\t{ref2}\n")
			fo.write(f"{label}\t{test1}\t{test2}\n")


def _pick_existing(path: str) -> str:
	if os.path.isfile(path):
		return path
	alt = path.replace('(1).wav', '.wav')
	if os.path.isfile(alt):
		return alt
	alt2 = path.replace('.wav', '(1).wav')
	if os.path.isfile(alt2):
		return alt2
	return path


def _list_wavs(root: str):
	for dirpath, _, filenames in os.walk(root):
		for fn in filenames:
			if fn.lower().endswith('.wav'):
				yield os.path.join(dirpath, fn)


_UTT_RE_1 = re.compile(r"_U(\d{3})_")
_UTT_RE_2 = re.compile(r"_U(\d{3})")


def _extract_utt_id(path: str):
	base = os.path.basename(path)
	m = _UTT_RE_1.search(base)
	if m:
		return m.group(1)
	m = _UTT_RE_2.search(base)
	if m:
		return m.group(1)
	return None


def _load_excluded_utt_ids_from_val_trials(val_trials_path: str):
	"""Extract utterance ids (U###) from the 5-column validation trial list."""
	excluded = set()
	with open(val_trials_path, 'r') as f:
		for line in f:
			parts = line.strip().split('\t')
			if len(parts) != 5:
				continue
			_, a1, a2, b1, b2 = parts
			for p in (a1, a2, b1, b2):
				utt = _extract_utt_id(p)
				if utt is not None:
					excluded.add(utt)
	return excluded


def _generate_train_list_from_dataset(
	bonafide_root: str,
	replay_root: str,
	out_path: str,
	seed: int = 0,
	max_bonafide: int = 0,
	max_replay: int = 0,
	exclude_utt_ids=None,
) -> None:
	"""Generate a 3-column training list: label<TAB>ref_wav<TAB>test_wav.

	This matches the *pair classification* that train_loader/train_network implement:
	- label 1: bonafide pair (SSN30 vs SSN80 from dataset/data_3_1_clip_16k)
	- label 0: replay pair (apple_1 vs apple_2 from dataset/replay/apple)

	max_* = 0 means no limit.
	"""
	rng = random.Random(seed)

	if exclude_utt_ids is None:
		exclude_utt_ids = set()

	# --- bonafide pairs: SSN30 vs SSN80 for same speaker+utterance ---
	bonafide_index = {}
	for wav in _list_wavs(bonafide_root):
		parts = wav.split(os.sep)
		# .../data_3_1_clip_16k/<SPK>/apple/<SSNxx>/file.wav
		try:
			spk = parts[parts.index('data_3_1_clip_16k') + 1]
			ssn = parts[parts.index('apple') + 1]
		except ValueError:
			continue
		utt = _extract_utt_id(wav)
		if utt is None:
			continue
		if utt in exclude_utt_ids:
			continue
		bonafide_index.setdefault(spk, {}).setdefault(utt, {})[ssn] = wav

	bonafide_pairs = []
	for spk, utts in bonafide_index.items():
		for utt, by_ssn in utts.items():
			p30 = by_ssn.get('SSN30')
			p80 = by_ssn.get('SSN80')
			if p30 and p80:
				bonafide_pairs.append((p30, p80))

	# --- replay pairs: apple_1 vs apple_2 for same speaker+utterance(+ssn when possible) ---
	def _apple_id(path: str):
		if f"{os.sep}apple_1{os.sep}" in path:
			return 'apple_1'
		if f"{os.sep}apple_2{os.sep}" in path:
			return 'apple_2'
		return None

	replay_index = {}
	for wav in _list_wavs(replay_root):
		parts = wav.split(os.sep)
		# .../replay/apple/apple_1/test3/<SPK>/<SSNxx>/file.wav
		if 'test3' not in parts:
			continue
		try:
			spk = parts[parts.index('test3') + 1]
		except ValueError:
			continue
		utt = _extract_utt_id(wav)
		if utt is None:
			continue
		if utt in exclude_utt_ids:
			continue
		ssn = None
		try:
			ssn = parts[parts.index(spk) + 1]
		except Exception:
			ssn = None
		app = _apple_id(wav)
		if app is None:
			continue
		replay_index.setdefault(spk, {}).setdefault(utt, {}).setdefault(ssn, {}).setdefault(app, []).append(wav)

	replay_pairs = []
	for spk, utts in replay_index.items():
		for utt, ssns in utts.items():
			for ssn, by_app in ssns.items():
				l1 = by_app.get('apple_1')
				l2 = by_app.get('apple_2')
				if not l1 or not l2:
					continue
				replay_pairs.append((rng.choice(l1), rng.choice(l2)))

	rng.shuffle(bonafide_pairs)
	rng.shuffle(replay_pairs)
	if max_bonafide and len(bonafide_pairs) > max_bonafide:
		bonafide_pairs = bonafide_pairs[:max_bonafide]
	if max_replay and len(replay_pairs) > max_replay:
		replay_pairs = replay_pairs[:max_replay]

	with open(out_path, 'w') as fo:
		for a, b in bonafide_pairs:
			fo.write(f"1\t{_pick_existing(a)}\t{_pick_existing(b)}\n")
		for a, b in replay_pairs:
			fo.write(f"0\t{_pick_existing(a)}\t{_pick_existing(b)}\n")


def _path_has_run_artifacts(path: str) -> bool:
	model_dir = os.path.join(path, 'model')
	if os.path.isdir(model_dir):
		try:
			for name in os.listdir(model_dir):
				if name.endswith('.model'):
					return True
		except OSError:
			return True
	if os.path.isfile(os.path.join(path, 'score.txt')):
		return True
	return False


def _make_unique_run_dir(base_path: str) -> str:
	if (not os.path.exists(base_path)):
		return base_path
	if (not _path_has_run_artifacts(base_path)):
		return base_path
	idx = 1
	while True:
		candidate = f"{base_path}_{idx:03d}"
		if not os.path.exists(candidate):
			return candidate
		if not _path_has_run_artifacts(candidate):
			return candidate
		idx += 1


parser = argparse.ArgumentParser(description="ECAPA_trainer")

parser.add_argument('--num_frames', type=int, default=200)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--n_cpu', type=int, default=1)
parser.add_argument('--device', type=str, default='mps')
parser.add_argument('--test_step', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay', type=float, default=0.97)

parser.add_argument('--train_path', type=str, default='')
parser.add_argument('--val_path', type=str, default='file_list/mel/dif/data_3_1/apple/val_apple_pair_list_replay_1_2.txt')
parser.add_argument('--save_path', type=str, default='results/data_3_1_apple_retrain')
parser.add_argument('--initial_model', type=str, default='')

parser.add_argument('--C', type=int, default=80)
parser.add_argument('--m', type=float, default=0.2)
parser.add_argument('--s', type=float, default=30)
parser.add_argument('--n_class', type=int, default=2)

parser.add_argument('--generate_train_from_pair_list', type=str, default='')
parser.add_argument('--generated_train_out', type=str, default='')

parser.add_argument('--generate_train_from_dataset', action='store_true')
parser.add_argument('--bonafide_root', type=str, default='dataset/data_3_1_clip_16k')
parser.add_argument('--replay_root', type=str, default='dataset/replay/apple')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--max_bonafide', type=int, default=0)
parser.add_argument('--max_replay', type=int, default=0)

parser.add_argument(	'--exclude_val_utt_ids',
	action='store_true',
	help='When generating a dataset-based train list, exclude any utterance IDs (U###) appearing in --val_path to avoid leakage.',
)


if __name__ == '__main__':
	warnings.simplefilter('ignore')
	args = parser.parse_args()

	args.device = _resolve_device(args.device)
	args.save_path = _make_unique_run_dir(args.save_path)

	if args.generate_train_from_pair_list:
		out_path = args.generated_train_out
		if not out_path:
			ts = time.strftime('%Y%m%d_%H%M%S')
			out_path = os.path.join(args.save_path, f'train_list_generated_{ts}.txt')
		if os.path.exists(out_path):
			root, ext = os.path.splitext(out_path)
			idx = 1
			while True:
				candidate = f"{root}_{idx:03d}{ext}"
				if not os.path.exists(candidate):
					out_path = candidate
					break
				idx += 1
		os.makedirs(os.path.dirname(out_path), exist_ok=True)
		_generate_train_list_from_pair_list(args.generate_train_from_pair_list, out_path)
		print(out_path)
		raise SystemExit(0)

	if args.generate_train_from_dataset:
		out_path = args.generated_train_out
		if not out_path:
			ts = time.strftime('%Y%m%d_%H%M%S')
			out_path = os.path.join(args.save_path, f'train_list_dataset_{ts}.txt')
		if os.path.exists(out_path):
			root, ext = os.path.splitext(out_path)
			idx = 1
			while True:
				candidate = f"{root}_{idx:03d}{ext}"
				if not os.path.exists(candidate):
					out_path = candidate
					break
				idx += 1
		os.makedirs(os.path.dirname(out_path), exist_ok=True)
		exclude_utt_ids = set()
		if args.exclude_val_utt_ids:
			if not args.val_path:
				raise ValueError('--exclude_val_utt_ids requires --val_path to be set')
			if not os.path.isfile(args.val_path):
				raise ValueError(f"--val_path does not exist: {args.val_path}")
			exclude_utt_ids = _load_excluded_utt_ids_from_val_trials(args.val_path)
		_generate_train_list_from_dataset(
			bonafide_root=args.bonafide_root,
			replay_root=args.replay_root,
			out_path=out_path,
			seed=args.seed,
			max_bonafide=args.max_bonafide,
			max_replay=args.max_replay,
			exclude_utt_ids=exclude_utt_ids,
		)
		print(out_path)
		raise SystemExit(0)

	if not args.train_path:
		raise ValueError('train_path is required for training. Provide a 3-column TSV: label<TAB>ref_wav<TAB>test_wav')
	if not os.path.isfile(args.train_path):
		raise ValueError(
			"train_path does not exist: %s\n"
			"First generate a training list with:\n"
			"  python trainECAPAModelL_dif_1.py --save_path <run_dir> --generate_train_from_pair_list %s\n"
			"Then re-run training using the printed path via --train_path." % (args.train_path, args.val_path)
		)

	args = init_args(args)

	dataset = train_loader(train_path=args.train_path)
	loader = torch.utils.data.DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.n_cpu,
		drop_last=True,
	)

	s = ECAPAModel(**vars(args))

	if args.initial_model and os.path.isfile(args.initial_model):
		print(f"Model {args.initial_model} loaded from previous state!")
		s.load_parameters(args.initial_model)

	for epoch in range(1, args.max_epoch + 1):
		loss, lr, acc = s.train_network(epoch=epoch, loader=loader)
		if epoch % args.test_step == 0:
			model_path = os.path.join(args.model_save_path, f'model_{epoch:04d}.model')
			s.save_parameters(model_path)
			print(model_path)
			if args.val_path:
				eer, mindcf = s.eval_network(val_path=args.val_path)
				print(f"EER {eer:2.2f}%")

