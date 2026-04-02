import argparse
import json
import os


def _hsv_to_rgb(h, s, v):
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(r * 255), int(g * 255), int(b * 255)


def _make_palette(n, seed=0.0):
    # evenly spaced hues
    palette = []
    for i in range(n):
        h = (i / max(1, n))
        r, g, b = _hsv_to_rgb(h, 0.85, 0.90)
        palette.append((r, g, b))
    return palette


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json_path', required=True)
    ap.add_argument('--out_path', default='')
    args = ap.parse_args()

    with open(args.json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cats = data.get('categories', [])
    if not cats:
        raise RuntimeError('No categories found in JSON')

    cats = sorted(cats, key=lambda c: c.get('id', 0))
    classes = [c.get('name', '').strip() for c in cats]
    palette = _make_palette(len(classes))

    lines = []
    lines.append('metainfo = dict(')
    lines.append('    classes=(')
    for name in classes:
        lines.append(f"        '{name}',")
    lines.append('    ),')
    lines.append('    palette=[')
    for color in palette:
        lines.append(f"        {color},")
    lines.append('    ]')
    lines.append(')')
    content = "\n".join(lines)

    if args.out_path:
        out_dir = os.path.dirname(args.out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_path, 'w', encoding='utf-8') as fo:
            fo.write(content)
        print('Wrote', args.out_path)
    else:
        print(content)


if __name__ == '__main__':
    main()
