#!/usr/bin/env python3
# dark variant of a Neutrino icon (for the dark mode of the system): python3 darkify.py light.svg dark.svg
# greys get their lightness inverted (black -> #e6e6e6, white -> #1f1f1f), saturated colours are kept
import re, sys, os, colorsys

# icons copied unchanged (colourful application logos; iconBW, the black logo of "Raise viewer", is converted)
# KEEP = {'icon', 'iconBW', 'logo', 'neutrino'}
KEEP = {'icon', 'logo', 'neutrino'}
# icons where the gradients (the light gear) are kept: they are already visible on a dark background
KEEP_GRADIENTS = {'gearPy', 'savePref', 'loadPref'}
# single colours kept (the flat light gear)
KEEP_COLORS = {'savePref': {'#cecece'}, 'loadPref': {'#cecece'}}

NAMED = {'black': '#000000', 'white': '#ffffff'}

def conv(h, keep=()):
    h = h.lower()
    if len(h) == 4:
        h = '#' + ''.join(c * 2 for c in h[1:])
    if h in keep:
        return h
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
    H, L, S = colorsys.rgb_to_hls(r, g, b)
    if S > 0.25 and 0.15 < L < 0.85:
        return h                                  # a real colour: keep
    L2 = 0.90 - 0.78 * L
    r, g, b = colorsys.hls_to_rgb(H, L2, S)
    return '#%02x%02x%02x' % tuple(round(x * 255) for x in (r, g, b))

def conv_colors(s, keep=()):
    s = re.sub(r'#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b', lambda m: conv(m.group(0), keep), s)
    s = re.sub(r'((?:fill|stroke|stop-color|color)\s*[:=]\s*"?)(black|white)\b',
               lambda m: m.group(1) + conv(NAMED[m.group(2)]), s)
    return s

def darkify(s, name):
    if name in KEEP:
        return s
    if name in KEEP_GRADIENTS:
        # convert everything except the gradient definitions
        parts = re.split(r'(<(?:linearGradient|radialGradient)\b.*?</(?:linearGradient|radialGradient)>)', s, flags=re.S)
        s = ''.join(p if i % 2 else conv_colors(p, KEEP_COLORS.get(name, ())) for i, p in enumerate(parts))
    else:
        s = conv_colors(s)
    # elements without fill are black by default (e.g. arrow markers): light default on the root, inherited
    s = re.sub(r'<svg\b(?![^>]*\sfill=)', '<svg fill="%s"' % conv('#000000'), s, count=1)
    return s

if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    name = os.path.splitext(os.path.basename(src))[0]
    with open(src) as f:
        s = f.read()
    with open(dst, 'w') as f:
        f.write(darkify(s, name))
