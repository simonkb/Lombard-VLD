import argparse
import os
import soundfile as sf


def _resample(data, orig_sr, target_sr):
    if orig_sr == target_sr:
        return data
    try:
        from scipy.signal import resample_poly
        import math
        g = math.gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        return resample_poly(data, up, down)
    except Exception:
        try:
            import resampy
            return resampy.resample(data, orig_sr, target_sr)
        except Exception as e:
            raise RuntimeError(
                'Resampling requires scipy or resampy. Install one of them.'
            ) from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_root', required=True)
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--target_sr', type=int, default=16000)
    ap.add_argument('--ext', default='.wav')
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    total = 0
    converted = 0
    for dirpath, _, filenames in os.walk(args.in_root):
        for fn in filenames:
            if not fn.lower().endswith(args.ext):
                continue
            total += 1
            in_path = os.path.join(dirpath, fn)
            rel = os.path.relpath(in_path, args.in_root)
            out_path = os.path.join(args.out_root, rel)
            out_dir = os.path.dirname(out_path)
            os.makedirs(out_dir, exist_ok=True)

            data, sr = sf.read(in_path)
            if sr != args.target_sr:
                data = _resample(data, sr, args.target_sr)
                converted += 1
            sf.write(out_path, data, args.target_sr, subtype='PCM_16')

    print('Total files:', total)
    print('Converted:', converted)
    print('Output root:', args.out_root)


if __name__ == '__main__':
    main()
