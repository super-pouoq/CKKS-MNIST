// ckks_demo.cu
// Phantom-FHE CKKS 接口演示：密钥生成 / 加密 / 解密 / 同态加法 / 同态乘法 /
// 密文矩阵-向量乘法（对角线编码 + 旋转法）。
//
// 编译运行见同目录 README.md。
//
// 关键概念：
//  - PhantomContext        : 由加密参数(EncryptionParameters)构造的上下文，所有运算都需要它
//  - PhantomSecretKey      : 私钥；可派生 PublicKey / RelinKey / GaloisKey；负责加密(对称)与解密
//  - PhantomPublicKey      : 公钥；负责非对称加密
//  - PhantomRelinKey       : 重线性化密钥；密文*密文后用来把密文从 3 项降回 2 项
//  - PhantomGaloisKey      : 伽罗瓦/旋转密钥；做 slot 旋转(rotate)时需要
//  - PhantomCKKSEncoder    : 把实数/复数向量 编码<->解码 成明文多项式(SIMD 打包)
//  - PhantomPlaintext / PhantomCiphertext : 明文 / 密文

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "phantom.h"

using namespace std;
using namespace phantom;
using namespace phantom::arith;

// 打印一个向量的前 n 个元素
static void print_vec(const string &tag, const vector<double> &v, size_t n = 8) {
    cout << tag << " [";
    n = min(n, v.size());
    for (size_t i = 0; i < n; i++) {
        cout << fixed << setprecision(4) << v[i];
        if (i + 1 < n) cout << ", ";
    }
    if (v.size() > n) cout << ", ...";
    cout << "]\n";
}

// ----------------------------------------------------------------------------
// 1) 构造 CKKS 上下文：选择多项式次数、模数链、scale
// ----------------------------------------------------------------------------
static PhantomContext make_ckks_context(size_t poly_modulus_degree,
                                        const vector<int> &mod_bits,
                                        size_t special_modulus_size = 1) {
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, mod_bits));
    parms.set_special_modulus_size(special_modulus_size);
    return PhantomContext(parms);
}

// ----------------------------------------------------------------------------
// 2) 加密 / 解密 演示
// ----------------------------------------------------------------------------
static void demo_enc_dec(PhantomContext &context, double scale) {
    cout << "\n========== [1] 加密 / 解密 ==========\n";

    // --- 密钥生成 ---
    PhantomSecretKey secret_key(context);                       // 私钥
    PhantomPublicKey public_key = secret_key.gen_publickey(context); // 公钥
    PhantomCKKSEncoder encoder(context);                        // 编码器

    size_t slot_count = encoder.slot_count();
    cout << "slot 数量 = " << slot_count << "\n";

    // --- 明文数据 ---
    vector<double> data(slot_count, 0.0);
    for (size_t i = 0; i < 8; i++) data[i] = 1.0 + i;  // 1,2,...,8
    print_vec("原始明文 :", data);

    // --- 编码 -> 明文多项式 ---
    PhantomPlaintext plain;
    encoder.encode(context, data, scale, plain);

    // --- 非对称加密(用公钥) ---
    PhantomCiphertext cipher;
    public_key.encrypt_asymmetric(context, plain, cipher);
    cout << "已用公钥加密。\n";

    // --- 解密(用私钥) ---
    PhantomPlaintext decrypted;
    secret_key.decrypt(context, cipher, decrypted);

    // --- 解码 -> 实数向量 ---
    vector<double> result;
    encoder.decode(context, decrypted, result);
    print_vec("解密结果 :", result);
}

// ----------------------------------------------------------------------------
// 3) 同态加法 / 乘法 演示
// ----------------------------------------------------------------------------
static void demo_add_mul(PhantomContext &context, double scale) {
    cout << "\n========== [2] 同态加法 / 乘法 ==========\n";

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);  // 乘法后重线性化需要
    PhantomCKKSEncoder encoder(context);

    size_t slot_count = encoder.slot_count();
    vector<double> a(slot_count, 0.0), b(slot_count, 0.0);
    for (size_t i = 0; i < 8; i++) { a[i] = i + 1; b[i] = 2.0; }   // a=1..8 , b=2

    PhantomPlaintext pa, pb;
    encoder.encode(context, a, scale, pa);
    encoder.encode(context, b, scale, pb);

    PhantomCiphertext ca, cb;
    public_key.encrypt_asymmetric(context, pa, ca);
    public_key.encrypt_asymmetric(context, pb, cb);

    // --- 密文加法: a + b ---
    PhantomCiphertext c_add = ca;
    add_inplace(context, c_add, cb);
    {
        PhantomPlaintext p; vector<double> r;
        secret_key.decrypt(context, c_add, p);
        encoder.decode(context, p, r);
        print_vec("a + b    :", r);   // 期望 3,4,...,10
    }

    // --- 密文乘法: a * b （乘后需 relinearize + rescale）---
    PhantomCiphertext c_mul = multiply(context, ca, cb);
    relinearize_inplace(context, c_mul, relin_keys); // 3 项 -> 2 项
    rescale_to_next_inplace(context, c_mul);         // 缩放回落，控制 scale 增长
    {
        PhantomPlaintext p; vector<double> r;
        secret_key.decrypt(context, c_mul, p);
        encoder.decode(context, p, r);
        print_vec("a * b    :", r);   // 期望 2,4,...,16
    }
}

// ----------------------------------------------------------------------------
// 4) 密文矩阵 × 明文向量： y = M * x   （M 为 d×d 明文方阵, x 为加密向量）
//
//    采用 CKKS 经典“对角线编码 + 旋转累加”算法 (Halevi-Shoup):
//      y = sum_{k=0}^{d-1}  diag_k(M) ⊙ rot(x, k)
//    其中 diag_k(M)[i] = M[i][(i+k) mod d]   是第 k 条(环绕)对角线，
//    rot(x,k) 把加密向量循环左移 k 位。
//    该方法只需 d 次 明文乘 + d-1 次旋转，全部在密文上完成。
// ----------------------------------------------------------------------------
static void demo_matrix_vector(PhantomContext &context, double scale) {
    cout << "\n========== [3] 密文矩阵 x 向量 (y = M*x) ==========\n";

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomGaloisKey galois_keys = secret_key.create_galois_keys(context); // 旋转需要
    PhantomCKKSEncoder encoder(context);

    size_t slot_count = encoder.slot_count();
    const size_t d = 4;  // 4x4 演示矩阵

    // 明文矩阵 M (行主序)
    vector<vector<double>> M = {
        {1, 2, 3, 4},
        {0, 1, 0, 1},
        {2, 0, 2, 0},
        {1, 1, 1, 1},
    };
    // 明文向量 x
    vector<double> x = {1, 2, 3, 4};

    cout << "矩阵 M:\n";
    for (auto &row : M) print_vec("   ", row, d);
    print_vec("向量 x:", x, d);

    // 期望结果（明文校验）: y[i] = sum_j M[i][j]*x[j]
    vector<double> expected(d, 0.0);
    for (size_t i = 0; i < d; i++)
        for (size_t j = 0; j < d; j++) expected[i] += M[i][j] * x[j];

    // --- 把 x 周期性平铺(tile)到所有 slot 后加密 ---
    // 关键: rotate_inplace 对整个 slot 空间做循环旋转。为了让长度为 d 的块内
    // 旋转“环绕”正确，需要把 x 以周期 d 重复填满所有 slot，
    // 这样全局循环左移 k 位 == 每个 d-块内循环左移 k 位。
    vector<double> x_slots(slot_count, 0.0);
    for (size_t i = 0; i < slot_count; i++) x_slots[i] = x[i % d];
    PhantomPlaintext px;
    encoder.encode(context, x_slots, scale, px);
    PhantomCiphertext cx;
    public_key.encrypt_asymmetric(context, px, cx);

    // --- 主循环：对每条对角线 k 做 (diag_k ⊙ rot(x,k)) 并累加 ---
    PhantomCiphertext acc;
    bool acc_init = false;

    for (size_t k = 0; k < d; k++) {
        // 第 k 条环绕对角线，同样周期平铺到所有 slot
        vector<double> diag(slot_count, 0.0);
        for (size_t i = 0; i < slot_count; i++) diag[i] = M[i % d][((i % d) + k) % d];

        PhantomPlaintext pdiag;
        encoder.encode(context, diag, scale, pdiag);

        // rot(x, k): 循环左移 k 位（k=0 时不旋转）
        PhantomCiphertext rx = cx;
        if (k != 0) rotate_inplace(context, rx, static_cast<int>(k), galois_keys);

        // diag_k ⊙ rot(x,k)
        multiply_plain_inplace(context, rx, pdiag);
        rescale_to_next_inplace(context, rx);

        if (!acc_init) {
            acc = rx;
            acc_init = true;
        } else {
            add_inplace(context, acc, rx);
        }
    }

    // --- 解密查看 y ---
    PhantomPlaintext py;
    secret_key.decrypt(context, acc, py);
    vector<double> y;
    encoder.decode(context, py, y);

    vector<double> y_head(y.begin(), y.begin() + d);
    print_vec("密文计算 y:", y_head, d);
    print_vec("明文期望 y:", expected, d);

    double max_err = 0.0;
    for (size_t i = 0; i < d; i++) max_err = max(max_err, fabs(y[i] - expected[i]));
    cout << "最大误差 = " << scientific << setprecision(3) << max_err << "\n";
}

int main() {
    cout << "Phantom-FHE CKKS 接口演示 (GPU 加速)\n";

    // 参数：N=2^14, 模数链 {60,40,40,40,60}（约 3 层乘法深度），scale=2^40
    const size_t poly_modulus_degree = 1 << 14;
    const vector<int> mod_bits = {60, 40, 40, 40, 60};
    const double scale = pow(2.0, 40);

    PhantomContext context = make_ckks_context(poly_modulus_degree, mod_bits, /*special=*/1);

    demo_enc_dec(context, scale);
    demo_add_mul(context, scale);
    demo_matrix_vector(context, scale);

    cout << "\n全部演示完成。\n";
    return 0;
}


