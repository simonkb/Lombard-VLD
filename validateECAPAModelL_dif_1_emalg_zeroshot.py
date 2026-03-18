import argparse
import os
import time
import warnings

import torch

from ECAPAModelL_dif_1_wav import ECAPAModel

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


parser = argparse.ArgumentParser(description='ECAPA EMALG zero-shot evaluation')
parser.add_argument('--device', type=str, default='mps')
parser.add_argument(
	'--initial_model',
	type=str,
	default='results/data_3_1_apple_retrain/model/model_0016.model',
)
parser.add_argument(
	'--emalg_roots',
	nargs='+',
	type=str,
	default=['enhanced_lombard_grid_part1', 'enhanced_lombard_grid_part2'],
)
parser.add_argument('--val_path', type=str, default='emalg_val_pair_list_plainplain_vs_lombardlombard.txt')
parser.add_argument('--generate_trials', action='store_true')
parser.add_argument('--num_trials', type=int, default=5000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save_dir', type=str, default='results/emalg_zeroshot')

parser.add_argument(
	'--within_speaker',
	action='store_true',
	help='If set, sample plain/plain and lombard/lombard pairs within the same speaker (reduces speaker leakage).',
)

parser.add_argument('--num_frames', type=int, default=200)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--n_cpu', type=int, default=1)
parser.add_argument('--test_step', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay', type=float, default=0.97)
parser.add_argument('--C', type=int, default=80)
parser.add_argument('--m', type=float, default=0.2)
parser.add_argument('--s', type=float, default=30)
parser.add_argument('--n_class', type=int, default=2)

parser.add_argument(
	'--confusion_matrix',
	action='store_true',
	default=False,
	help='If set, also run eval_network_confusion_matrix (slower) and print its accuracy.',
)


if __name__ == '__main__':
	warnings.simplefilter('ignore')
	args = parser.parse_args()

	args.device = _resolve_device(args.device)

	os.makedirs(args.save_dir, exist_ok=True)

	val_path = args.val_path

	if not os.path.isfile(args.initial_model):
		raise ValueError(f"initial_model does not exist: {args.initial_model}")
	if not os.path.isfile(val_path):
		raise ValueError(f"val_path does not exist: {val_path}")

	s = ECAPAModel(**vars(args))
	print(f"Model {args.initial_model} loaded from previous state!")
	s.load_parameters(args.initial_model)
	eer, mindcf = s.eval_network(val_path=val_path)
	print("EER %2.2f%%, minDCF %.4f%%" % (eer, mindcf))

	if args.confusion_matrix:
		# eval_network_confusion_matrix expects a 3-column file list, while EMALG
		# trials are 5-column. Skip to avoid crashing.
		with open(val_path, 'r') as f:
			first = f.readline().rstrip('\n')
		if first.count('\t') >= 4:
			print('Skipping confusion-matrix evaluation: val_path appears to be a 5-column trial list.')
		else:
			accuracy1 = s.eval_network_confusion_matrix(val_path=val_path)
			print(accuracy1)
