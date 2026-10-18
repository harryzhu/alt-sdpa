#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <stdexcept>
#include <string>
#include <cmath>
#include <algorithm>
#include <vector>

#include <mlx/mlx.h>

// 跨平台BLAS头文件
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

namespace nb = nanobind;


// 辅助函数：对单个向量进行 Softmax
void softmax_vector(float* data, size_t len) {
    if (len == 0) return;

    // 1. 找最大值 (防止 exp 溢出)
    float max_val = data[0];
    for (size_t i = 1; i < len; ++i) {
        if (data[i] > max_val){
            max_val = data[i];
        }
    }

    // 2. 计算 exp(x - max) 并求和
    float sum = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        data[i] = std::exp(data[i] - max_val);
        sum += data[i];
    }

    // 3. 归一化
    float inv_sum = 1.0f / sum;
    
    // 使用 vDSP 加速乘法 (macOS) 或普通循环
    #ifdef __APPLE__
        vDSP_vsmul(data, 1, &inv_sum, data, 1, len);
    #else
        for (size_t i = 0; i < len; ++i) {
            data[i] *= inv_sum;
        }
    #endif
}

void softmax_4d(
    nb::ndarray<nb::numpy, float> x // Shape: [B, H, L, L]
) {
    if (x.ndim() != 4) {
        throw std::invalid_argument("Inputs must be 4D");
    }
    
    size_t B = x.shape(0);
    size_t H = x.shape(1);
    size_t L = x.shape(2);
    size_t Dh = x.shape(3);

    if(L != Dh){
        throw std::runtime_error("weights shape mismatch: L should equal Dh");
    }

    float* ptr_x = x.data();

    // 总共有 B * H * L 个向量需要处理
    size_t num_vectors = B * H * L;
    size_t vector_len = L;

    // 2. 对每个向量执行 Softmax
    for (size_t i = 0; i < num_vectors; ++i) {
        float* vec_ptr = ptr_x + i * vector_len;
        softmax_vector(vec_ptr, vector_len);
    }
}

void sdpa_cpu(
    nb::ndarray<nb::numpy, const float> q,
    nb::ndarray<nb::numpy, const float> kt,
    nb::ndarray<nb::numpy, const float> v,
    nb::ndarray<nb::numpy, const float> mask,
    nb::ndarray<nb::numpy, float> scores,
    nb::ndarray<nb::numpy, float> out_atten
) {
    // 维度校验
    if (q.ndim() != 4 || kt.ndim() != 4 || v.ndim() != 4 || mask.ndim() != 4) {
        throw std::runtime_error("q,k,v,mask must be 4D matrices");
    }
    size_t B_q = q.shape(0);
    size_t H_q = q.shape(1);
    size_t L_q = q.shape(2);
    size_t Dh_q = q.shape(3);
    //
    size_t B_kt = kt.shape(0);
    size_t H_kt = kt.shape(1);
    size_t L_kt = kt.shape(3);
    size_t Dh_kt = kt.shape(2);
    //
    size_t B_v = v.shape(0);
    size_t H_v = v.shape(1);
    size_t L_v = v.shape(2);
    size_t Dh_v = v.shape(3);
    //
    // out: B, H, L, L
    size_t B_out = scores.shape(0);
    size_t H_out = scores.shape(1);
    size_t L_out = scores.shape(3);
    size_t L2_out = scores.shape(2);
    //
    size_t B_m = mask.shape(0);
    size_t H_m = mask.shape(1);
    size_t L_m = mask.shape(2);
    size_t L2_m = mask.shape(3);
    //
    size_t B_atten = out_atten.shape(0);
    size_t H_atten = out_atten.shape(1);
    size_t L_atten = out_atten.shape(2);
    size_t Dh_atten = out_atten.shape(3);

    if(B_q != B_kt || B_q != B_v || B_q !=B_m || B_q != B_atten || B_q != B_out || B_kt != B_out){
        throw std::runtime_error("query, key, value shape mismatch: batch_size");
    }

    if(H_q != H_kt || H_q != H_v || H_q != H_out || H_kt != H_out || H_q != H_atten){
        throw std::runtime_error("query, key shape mismatch: num_head");
    }

    if(Dh_q != Dh_kt || Dh_q != Dh_v){
        throw std::runtime_error("query, key shape mismatch: head_dim");
    }

    if(L_q != L_out || L_kt != L_out ){
        throw std::runtime_error("query, key, out shape mismatch: Lseq");
    }

    if(L_v != L_out || Dh_v != Dh_q ){
        throw std::runtime_error("val, out shape mismatch: Lseq, Dh");
    }
    
    if (L_out != L2_out) {
        throw std::runtime_error("Output shape mismatch: out");
    }

    if (H_m != 1) {
        throw std::runtime_error("mask shape mismatch: mask[1] should be 1");
    }

    if (L_m != L2_m || L_m != L_q) {
        throw std::runtime_error("mask shape mismatch: mask[2 and 3] should be L");
    }
    
    // 零拷贝：直接拿numpy底层内存指针
    const float* ptr_q = q.data();
    const float* ptr_k = kt.data();
    float* ptr_out = scores.data();

    const float alpha_Dh = 1.0 / sqrt(Dh_q);

    for (size_t i = 0; i < B_q * H_q; ++i) {
        const float* q_slice = ptr_q + i * (L_q * Dh_q);
        const float* k_slice = ptr_k + i * (L_kt * Dh_kt);
        float* out_slice = ptr_out + i * (L_out * L2_out);

        // 3. 调用 cblas_sgemm
        // C = alpha * A * B + beta * C
        // A is [L, Dh], B is [Dh, L_out], C is [L, L_out]

        cblas_sgemm(
            CblasRowMajor,    // 行优先存储 (NumPy 默认)
            CblasNoTrans,     // A 不转置
            CblasNoTrans,     // B 不转置
            L_q,                // M: A 的行数
            L_out,            // N: B 的列数
            Dh_q,               // K: A 的列数 / B 的行数
            alpha_Dh,             // alpha
            q_slice,          // A 数据指针
            Dh_q,               // lda: A 的行跨度 (因为行优先，lda = K)
            k_slice,          // B 数据指针
            L_out,            // ldb: B 的行跨度 (因为行优先，ldb = N)
            0.0f,             // beta (覆盖输出)
            out_slice,        // C 数据指针
            L_out             // ldc: C 的行跨度 (因为行优先，ldc = N)
        );

    } // q@kt/sqrt(Dh)

    const float* ptr_m = mask.data();
    size_t matrix_size = L_m * L2_m;
    for (size_t i = 0; i < B_m; ++i) {
        float* score_batch = ptr_out + i * (H_out * matrix_size);
        const float* m_batch = ptr_m + i * (1 * matrix_size); // b 在 H 维只有 1
        
        for (size_t j = 0; j < H_out; ++j) {
            // 获取当前 Head 的矩阵指针
            const float* score_mat = score_batch + j * matrix_size;
            const float* m_mat = m_batch; // 注意：b_mat 不随 h 变化！这就是广播
            float* out_mat = score_batch + j * matrix_size;

            // 3. 逐元素相加 (编译器会自动向量化优化)
            for (size_t n = 0; n < matrix_size; ++n) {
                out_mat[n] = score_mat[n] + m_mat[n];
            }
        }
    } // score + mask

    softmax_4d(scores); // softmax

    // atten: attn_out = softmax_ed.transpose(0,2,1,3) @ v
    const float* ptr_v = v.data();
    float* ptr_atten = out_atten.data();
    for (size_t i = 0; i < B_v * H_v; ++i) {
        const float* w_slice = ptr_out + i * (L_out * L2_out);
        const float* v_slice = ptr_v + i * (L_v * Dh_v);
        float* atten_slice = ptr_atten + i * (L_atten * Dh_atten);

        cblas_sgemm(
            CblasRowMajor,    // 行优先存储 (NumPy 默认)
            CblasNoTrans,     // A 不转置
            CblasNoTrans,     // B 不转置
            L_out,                // M: A 的行数
            Dh_v,            // N: B 的列数
            L_v,               // K: A 的列数 / B 的行数
            1.0f,             // alpha
            w_slice,          // A 数据指针
            L2_out,               // lda: A 的行跨度 (因为行优先，lda = K)
            v_slice,          // B 数据指针
            Dh_v,            // ldb: B 的行跨度 (因为行优先，ldb = N)
            0.0f,             // beta (覆盖输出)
            atten_slice,        // C 数据指针
            Dh_atten             // ldc: C 的行跨度 (因为行优先，ldc = N)
        );

        //最后的转置、reshape，需要在python里面做
        //attn_out.transpose(0,2,1,3).reshape(B, L, D)
    }   
}



nb::object sdpa_gpu(
    nb::object q, 
    nb::object kt, 
    nb::object v,
    nb::object mask) {
    // 通过 Python 调用 mlx.matmul，保证使用同一个 MLX 实例
    auto mx = nb::module_::import_("mlx.core");
    auto fn_matmul = mx.attr("matmul");
    auto fn_multiply = mx.attr("multiply");
    auto fn_add = mx.attr("add");
    auto fn_softmax = mx.attr("softmax");
    size_t Dh_q = nb::cast<size_t>(q.attr("shape").attr("__getitem__")(nb::int_(-1)));
    
    float alpha = 1.0/sqrt(Dh_q);
    auto qkt = fn_matmul(q,kt);
    auto scores = fn_multiply(qkt, alpha);
    auto atten_scores = fn_add(scores, mask);
    auto weights = fn_softmax(atten_scores, nb::int_(-1));

    nb::object out_atten = fn_matmul(weights, v);
    return out_atten;
}

NB_MODULE(alt_sdpa_ext, m) {
    nb::module_::import_("mlx.core");

    m.doc() = R"(alternative Scaled-Dot-Product-Attention function for Transformer's Decoder;
    - Support combined mask (additive mask + padding mask) on MLX;
    - because mlx's 'mx.fast.scaled_dot_product_attention' does not support combined mask, 
    that's why I made this.)";
    
    m.def("sdpa_gpu", &sdpa_gpu,
          nb::arg("q"), nb::arg("kt"), nb::arg("v"), nb::arg("mask"),
          "sdpa_gpu is using mlx");
        
    m.def("sdpa_cpu", &sdpa_cpu,
          nb::arg("q"), nb::arg("kt"), nb::arg("v"), nb::arg("mask"), nb::arg("scores"), nb::arg("out_atten"),
          R"(parameters: 
            1.输入的矩阵必须内存连续: x = np.ascontiguousarray(x);
            2.scores和out_atten是为了计算预分配的空矩阵,形状必须满足下面的要求;
            3.out_atten是最终的输出结果,接收以后,需要在python做一下transpose和reshape:
            attention = out_atten.transpose(0,2,1,3).reshape(B, L, H*head_dim)
            #
            q,k,v: shape[B,H,L,head_dim], 分别做一次np.ascontiguousarray();
            mask: shape[B,1,L,L], np.ascontiguousarray(mask);
            scores: np.ascontiguousarray(np.empty(shape=[B,H,L,L], dtype=np.float32));
            out_atten: np.ascontiguousarray(np.empty(shape=[B,H,L,head_dim], dtype=np.float32));
        )"
    );
}
