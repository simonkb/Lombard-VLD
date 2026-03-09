import argparse, os, time, warnings

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

