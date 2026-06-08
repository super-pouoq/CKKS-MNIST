// ckks_mnist.cu
// ---------------------------------------------------------------------------
// Phantom-FHE 上的 MNIST 加密推理 (CKKS)。
//
// 明文模型 (../../MNIST/normal/model.py, 已训练, 用 x^2 替 ReLU / AvgPool 替 MaxPool):
//   conv1 Conv2d(1,16,3,pad=1) -> square -> AvgPool2d(2)   => 16 x 14 x 14
//   conv2 Conv2d(16,32,3,pad=1)-> square -> AvgPool2d(2)   => 32 x 7  x 7
//   flatten(1568) -> fc1 Linear(1568,128) -> square -> fc2 Linear(128,10)
//
// 打包: channel-packing, 每个特征图通道 = 一个密文; 像素 (y,x) 放在 slot
//       y*step_row + x*step_col。池化用 rotate+add(不乘), 让有效值留在 stride
//       布局上(step 翻倍), /4 折进下一层权重, 从而把乘法深度压到最小。
//
// 乘法/rescale 深度(关键路径):
//   conv1(1) square(1) conv2(1) square(1) fc1(mul1+mask1=2) square(1) fc2(2) = 9。
//   故取 N=2^15, 模数链 {60, 40*9, 60}=480 bit (< 2^15 在 128-bit 安全下的上限 881)。
// ---------------------------------------------------------------------------

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "phantom.h"

using namespace std;
using namespace phantom;
using namespace phantom::arith;

// ============================= 读权重/图片 =============================
struct Tensor {
    vector<int> shape;
    vector<double> data;
    int numel() const { int n = 1; for (int s : shape) n *= s; return n; }
};

static map<string, Tensor> load_weights(const string &path) {
    ifstream f(path);
    if (!f) { cerr << "cannot open " << path << "\n"; exit(1); }
    map<string, Tensor> w;
    string line;
    while (getline(f, line)) {
        if (line.empty()) continue;
        istringstream hs(line);
        string name; hs >> name;
        Tensor t; int s;
        while (hs >> s) t.shape.push_back(s);
        string vals; getline(f, vals);
        istringstream vs(vals);
        t.data.reserve(t.numel());
        double v;
        while (vs >> v) t.data.push_back(v);
        w[name] = std::move(t);
    }
    return w;
}

struct ImageSet {
    int count, rows, cols;
    vector<int> labels;
    vector<vector<double>> pixels;
};

static ImageSet load_images(const string &path) {
    ifstream f(path);
    if (!f) { cerr << "cannot open " << path << "\n"; exit(1); }
    ImageSet s;
    f >> s.count >> s.rows >> s.cols;
    s.labels.resize(s.count);
    s.pixels.resize(s.count);
    int n = s.rows * s.cols;
    for (int i = 0; i < s.count; i++) {
        f >> s.labels[i];
        s.pixels[i].resize(n);
        for (int j = 0; j < n; j++) f >> s.pixels[i][j];
    }
    return s;
}

// ============================ CKKS 包装 ============================
struct FHE {
    PhantomContext *ctx;
    PhantomSecretKey *sk;
    PhantomPublicKey *pk;
    PhantomRelinKey *rlk;
    PhantomGaloisKey *glk;
    PhantomCKKSEncoder *enc;
    double scale;
    size_t slots;
};

static void mul_const_vec(FHE &fhe, PhantomCiphertext &ct, const vector<double> &vals) {
    PhantomPlaintext p;
    fhe.enc->encode(*fhe.ctx, vals, ct.scale(), p, ct.chain_index());
    multiply_plain_inplace(*fhe.ctx, ct, p);
    rescale_to_next_inplace(*fhe.ctx, ct);
}

static void square_ct(FHE &fhe, PhantomCiphertext &ct) {
    PhantomCiphertext t = multiply(*fhe.ctx, ct, ct);
    relinearize_inplace(*fhe.ctx, t, *fhe.rlk);
    rescale_to_next_inplace(*fhe.ctx, t);
    ct = t;
}

static void add_aligned(FHE &fhe, PhantomCiphertext &acc, PhantomCiphertext src) {
    if (acc.chain_index() < src.chain_index())
        mod_switch_to_inplace(*fhe.ctx, src, acc.chain_index());
    else if (src.chain_index() < acc.chain_index())
        mod_switch_to_inplace(*fhe.ctx, acc, src.chain_index());
    acc.set_scale(src.scale());
    add_inplace(*fhe.ctx, acc, src);
}

static void add_const_vec(FHE &fhe, PhantomCiphertext &ct, const vector<double> &vals) {
    PhantomPlaintext p;
    fhe.enc->encode(*fhe.ctx, vals, ct.scale(), p, ct.chain_index());
    add_plain_inplace(*fhe.ctx, ct, p);
}

// ============================ 卷积 ============================
// 像素 (y,x) 存于 slot y*step_row + x*step_col, y,x∈[0,H),[0,W)。
// 3x3 卷积按 tap 平移; shift = dy*step_row + dx*step_col。边界 mask 与权重折进系数。
// fold: 折叠进权重的标量(如池化 /4)。输出布局与输入相同。
static vector<PhantomCiphertext> conv2d(
        FHE &fhe, const vector<PhantomCiphertext> &in_cts,
        const Tensor &weight, const Tensor &bias,
        int H, int W, int step_row, int step_col,
        double fold = 1.0) {
    int C_out = weight.shape[0];
    int C_in = weight.shape[1];
    int K = weight.shape[2];
    int pad = K / 2;

    vector<vector<char>> valid(K * K, vector<char>(H * W, 0));
    vector<int> shifts(K * K);
    for (int ky = 0; ky < K; ky++)
        for (int kx = 0; kx < K; kx++) {
            int dy = ky - pad, dx = kx - pad;
            shifts[ky * K + kx] = dy * step_row + dx * step_col;
            auto &m = valid[ky * K + kx];
            for (int oy = 0; oy < H; oy++)
                for (int ox = 0; ox < W; ox++) {
                    int iy = oy + dy, ix = ox + dx;
                    if (iy >= 0 && iy < H && ix >= 0 && ix < W) m[oy * W + ox] = 1;
                }
        }

    vector<PhantomCiphertext> out_cts(C_out);
    for (int oc = 0; oc < C_out; oc++) {
        bool init = false;
        PhantomCiphertext acc;
        for (int ic = 0; ic < C_in; ic++) {
            for (int t = 0; t < K * K; t++) {
                double wv = weight.data[((oc * C_in + ic) * K * K) + t] * fold;
                if (wv == 0.0) continue;
                PhantomCiphertext r = in_cts[ic];
                if (shifts[t] != 0) rotate_inplace(*fhe.ctx, r, shifts[t], *fhe.glk);
                vector<double> coef(fhe.slots, 0.0);
                for (int oy = 0; oy < H; oy++)
                    for (int ox = 0; ox < W; ox++)
                        if (valid[t][oy * W + ox]) coef[oy * step_row + ox * step_col] = wv;
                mul_const_vec(fhe, r, coef);
                if (!init) { acc = r; init = true; }
                else add_aligned(fhe, acc, r);
            }
        }
        vector<double> bvec(fhe.slots, 0.0);
        for (int oy = 0; oy < H; oy++)
            for (int ox = 0; ox < W; ox++)
                bvec[oy * step_row + ox * step_col] = bias.data[oc];
        add_const_vec(fhe, acc, bvec);
        out_cts[oc] = acc;
    }
    return out_cts;
}

// ============================ AvgPool2d(2) ============================
// 仅 rotate+add 求 2x2 之和(不乘 /4); 输出有效值落在 (2oy,2ox), 布局步长翻倍。
static vector<PhantomCiphertext> avgpool2_sum(
        FHE &fhe, const vector<PhantomCiphertext> &in_cts, int step_row, int step_col) {
    vector<PhantomCiphertext> out(in_cts.size());
    for (size_t c = 0; c < in_cts.size(); c++) {
        PhantomCiphertext s = in_cts[c];
        PhantomCiphertext r1 = in_cts[c]; rotate_inplace(*fhe.ctx, r1, step_col, *fhe.glk);
        PhantomCiphertext r2 = in_cts[c]; rotate_inplace(*fhe.ctx, r2, step_row, *fhe.glk);
        PhantomCiphertext r3 = in_cts[c]; rotate_inplace(*fhe.ctx, r3, step_row + step_col, *fhe.glk);
        add_inplace(*fhe.ctx, s, r1);
        add_inplace(*fhe.ctx, s, r2);
        add_inplace(*fhe.ctx, s, r3);
        out[c] = s;
    }
    return out;
}

// ============================ 全连接层 ============================
// out[o] = sum_{c,k} W[o, c*plane+k] * x_at(positions[c][k]); 深度 2(权重乘 + 掩码)。
// bias 在外部加。结果落在 slot 0..out_f-1。
static PhantomCiphertext fc_layer(
        FHE &fhe, const vector<PhantomCiphertext> &in_cts,
        const vector<vector<int>> &positions, int plane, const Tensor &weight) {
    int out_f = weight.shape[0];
    int C = in_cts.size();

    PhantomCiphertext result;
    bool result_init = false;
    for (int o = 0; o < out_f; o++) {
        PhantomCiphertext acc;
        bool init = false;
        for (int c = 0; c < C; c++) {
            vector<double> coef(fhe.slots, 0.0);
            int base = o * (C * plane) + c * plane;
            for (int k = 0; k < plane; k++) coef[positions[c][k]] = weight.data[base + k];
            PhantomCiphertext piece = in_cts[c];
            mul_const_vec(fhe, piece, coef);
            if (!init) { acc = piece; init = true; }
            else add_aligned(fhe, acc, piece);
        }
        int step = 1;
        while ((size_t)step < fhe.slots) {
            PhantomCiphertext rr = acc;
            rotate_inplace(*fhe.ctx, rr, step, *fhe.glk);
            add_inplace(*fhe.ctx, acc, rr);
            step <<= 1;
        }
        vector<double> sel(fhe.slots, 0.0);
        sel[o] = 1.0;
        mul_const_vec(fhe, acc, sel);
        if (!result_init) { result = acc; result_init = true; }
        else add_aligned(fhe, result, acc);
    }
    return result;
}

int main(int argc, char **argv) {
    string data_dir = (argc > 1) ? argv[1] : "data";
    int max_imgs = (argc > 2) ? atoi(argv[2]) : 5;

    cout << "Phantom-FHE CKKS MNIST 加密推理\n";
    auto W = load_weights(data_dir + "/weights.txt");
    auto imgs = load_images(data_dir + "/images.txt");
    cout << "weights/images loaded. images=" << imgs.count
         << " (" << imgs.rows << "x" << imgs.cols << ")\n";

    const size_t N = 1 << 15;
    vector<int> mod_bits = {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60};  // 9 mid levels
    const double scale = pow(2.0, 40);

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, mod_bits));
    parms.set_special_modulus_size(1);
    PhantomContext context(parms);

    PhantomSecretKey sk(context);
    PhantomPublicKey pk = sk.gen_publickey(context);
    PhantomRelinKey rlk = sk.gen_relinkey(context);
    PhantomGaloisKey glk = sk.create_galois_keys(context);
    PhantomCKKSEncoder encoder(context);

    FHE fhe{&context, &sk, &pk, &rlk, &glk, &encoder, scale, encoder.slot_count()};
    cout << "slot_count=" << fhe.slots << ", mid levels=" << (mod_bits.size() - 2) << "\n";

    int n = min(max_imgs, imgs.count);
    int correct = 0;
    for (int i = 0; i < n; i++) {
        // 输入: 28x28, step_row=28, step_col=1
        vector<double> slotbuf(fhe.slots, 0.0);
        for (int p = 0; p < imgs.rows * imgs.cols; p++) slotbuf[p] = imgs.pixels[i][p];
        PhantomPlaintext pin;
        encoder.encode(context, slotbuf, scale, pin);
        PhantomCiphertext cin;
        pk.encrypt_asymmetric(context, pin, cin);
        vector<PhantomCiphertext> x = {cin};

        // conv1 -> square -> pool
        x = conv2d(fhe, x, W.at("conv1.weight"), W.at("conv1.bias"), 28, 28, 28, 1);
        for (auto &c : x) square_ct(fhe, c);
        x = avgpool2_sum(fhe, x, 28, 1);                  // step -> (56, 2), 14x14

        // conv2(把 /4 折进权重) -> square -> pool
        x = conv2d(fhe, x, W.at("conv2.weight"), W.at("conv2.bias"), 14, 14, 56, 2, 0.25);
        for (auto &c : x) square_ct(fhe, c);
        x = avgpool2_sum(fhe, x, 56, 2);                  // step -> (112, 4), 7x7

        // fc1: 32 通道 x 49; 位置 (oy,ox)->slot oy*112+ox*4; /4 折进权重
        int plane1 = 49;
        vector<vector<int>> pos1(32, vector<int>(plane1));
        for (int oy = 0; oy < 7; oy++)
            for (int ox = 0; ox < 7; ox++)
                for (int c = 0; c < 32; c++)
                    pos1[c][oy * 7 + ox] = oy * 112 + ox * 4;
        Tensor fc1w = W.at("fc1.weight");
        for (auto &v : fc1w.data) v *= 0.25;
        PhantomCiphertext h1 = fc_layer(fhe, x, pos1, plane1, fc1w);
        {
            vector<double> bvec(fhe.slots, 0.0);
            for (int o = 0; o < 128; o++) bvec[o] = W.at("fc1.bias").data[o];
            add_const_vec(fhe, h1, bvec);
        }
        square_ct(fhe, h1);

        // fc2: 输入 128 个在 slot 0..127
        vector<vector<int>> pos2(1, vector<int>(128));
        for (int k = 0; k < 128; k++) pos2[0][k] = k;
        PhantomCiphertext out = fc_layer(fhe, {h1}, pos2, 128, W.at("fc2.weight"));
        {
            vector<double> bvec(fhe.slots, 0.0);
            for (int o = 0; o < 10; o++) bvec[o] = W.at("fc2.bias").data[o];
            add_const_vec(fhe, out, bvec);
        }

        PhantomPlaintext pout;
        sk.decrypt(context, out, pout);
        vector<double> dec;
        encoder.decode(context, pout, dec);

        int pred = 0;
        for (int j = 1; j < 10; j++) if (dec[j] > dec[pred]) pred = j;
        correct += (pred == imgs.labels[i]);
        cout << "img" << i << ": label=" << imgs.labels[i] << " pred=" << pred << "  logits=[";
        for (int j = 0; j < 10; j++) { cout << fixed << setprecision(2) << dec[j]; if (j < 9) cout << " "; }
        cout << "]\n";
    }
    cout << "encrypted acc: " << correct << "/" << n << "\n";
    return 0;
}
