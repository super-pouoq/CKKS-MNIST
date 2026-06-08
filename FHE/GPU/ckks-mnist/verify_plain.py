#!/usr/bin/env python3
# verify_plain.py
# 纯 Python(无 torch) 复现 model.py 的前向，验证导出的 weights.txt / images.txt 正确。
# 网络: conv1->square->avgpool2 -> conv2->square->avgpool2 -> flatten -> fc1->square -> fc2
import sys

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"

def load_weights(path):
    w = {}
    with open(path) as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        head = lines[i].split()
        name = head[0]
        shape = tuple(int(x) for x in head[1:])
        vals = [float(x) for x in lines[i+1].split()]
        w[name] = (shape, vals)
        i += 2
    return w

def load_images(path):
    with open(path) as f:
        lines = f.read().splitlines()
    k, rows, cols = (int(x) for x in lines[0].split())
    imgs = []
    idx = 1
    for _ in range(k):
        label = int(lines[idx]); idx += 1
        px = [float(x) for x in lines[idx].split()]; idx += 1
        imgs.append((label, px))
    return rows, cols, imgs

def conv2d(inp, C_in, H, W, weight, bias, C_out, K=3, pad=1):
    # inp: list of C_in*H*W; weight: [C_out,C_in,K,K]; out HxW same (pad=1,stride1)
    out = [0.0] * (C_out * H * W)
    for oc in range(C_out):
        b = bias[oc]
        for oy in range(H):
            for ox in range(W):
                acc = b
                for ic in range(C_in):
                    for ky in range(K):
                        iy = oy + ky - pad
                        if iy < 0 or iy >= H: continue
                        for kx in range(K):
                            ix = ox + kx - pad
                            if ix < 0 or ix >= W: continue
                            wv = weight[((oc*C_in+ic)*K+ky)*K+kx]
                            acc += wv * inp[(ic*H+iy)*W+ix]
                out[(oc*H+oy)*W+ox] = acc
    return out

def square(x): return [v*v for v in x]

def avgpool2(inp, C, H, W):
    OH, OW = H//2, W//2
    out = [0.0]*(C*OH*OW)
    for c in range(C):
        for oy in range(OH):
            for ox in range(OW):
                s = (inp[(c*H+2*oy)*W+2*ox] + inp[(c*H+2*oy)*W+2*ox+1]
                     + inp[(c*H+2*oy+1)*W+2*ox] + inp[(c*H+2*oy+1)*W+2*ox+1])
                out[(c*OH+oy)*OW+ox] = s/4.0
    return out

def linear(inp, weight, bias, out_f, in_f):
    out = [0.0]*out_f
    for o in range(out_f):
        acc = bias[o]
        base = o*in_f
        for i in range(in_f):
            acc += weight[base+i]*inp[i]
        out[o] = acc
    return out

def forward(px, w):
    c1w = w["conv1.weight"][1]; c1b = w["conv1.bias"][1]
    c2w = w["conv2.weight"][1]; c2b = w["conv2.bias"][1]
    f1w = w["fc1.weight"][1]; f1b = w["fc1.bias"][1]
    f2w = w["fc2.weight"][1]; f2b = w["fc2.bias"][1]
    x = conv2d(px, 1, 28, 28, c1w, c1b, 16); x = square(x); x = avgpool2(x, 16, 28, 28)  # 16x14x14
    x = conv2d(x, 16, 14, 14, c2w, c2b, 32); x = square(x); x = avgpool2(x, 32, 14, 14)   # 32x7x7
    x = square(linear(x, f1w, f1b, 128, 32*7*7))
    x = linear(x, f2w, f2b, 10, 128)
    return x

def main():
    w = load_weights(f"{DATA}/weights.txt")
    rows, cols, imgs = load_images(f"{DATA}/images.txt")
    correct = 0
    for n,(label,px) in enumerate(imgs):
        logits = forward(px, w)
        pred = max(range(10), key=lambda i: logits[i])
        correct += (pred == label)
        if n < 10:
            print(f"img{n}: label={label} pred={pred} logit={logits[pred]:.3f}")
    print(f"plaintext acc on {len(imgs)}: {correct}/{len(imgs)}")

if __name__ == "__main__":
    main()
