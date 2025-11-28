#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define EGGROLL_USE_NEON 1
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#define EGGROLL_USE_AVX2 1
#endif

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#else
#include <pthread.h>
#include <unistd.h>
#define EGGROLL_USE_PTHREADS 1
#endif

#if defined(EGGROLL_USE_NEON)
#define EGGROLL_SIMD_NAME "ARM NEON"
#elif defined(EGGROLL_USE_AVX2)
#define EGGROLL_SIMD_NAME "x86 AVX2"
#else
#define EGGROLL_SIMD_NAME "Scalar"
#endif

#if defined(__APPLE__)
#define EGGROLL_PARALLEL_BACKEND "Grand Central Dispatch"
#elif defined(EGGROLL_USE_PTHREADS)
#define EGGROLL_PARALLEL_BACKEND "Pthreads"
#else
#define EGGROLL_PARALLEL_BACKEND "Serial loop"
#endif

// --- Configuration [cite: 275, 277, 288] ---
#define VOCAB_SIZE 256        // Byte-level tokenization
#define HIDDEN_DIM 128        // Model width
#define N_LAYERS 2            // Number of layers
#define SEQ_LEN 512            // Sequence length for BPTT (truncated)
#define POPULATION_SIZE 16    // Number of perturbations per step
#define BATCH_SIZE 2          // Parallel streams
#define FIXED_POINT 4         // 4 bits for fractional part
#define SIGMA_SHIFT 4         // Noise scale (bitwise shift)
#define UPDATE_THRESHOLD 160  // Votes needed to flip a weight [cite: 1023]
#define MAX_VAL 127
#define MIN_VAL -127

// Derived Constants for Optimization
#define MAX_MATRIX_DIM (HIDDEN_DIM * 4) // Largest dimension (MLP expansion)
#define MAX_POP_PAIRS (POPULATION_SIZE / 2)

// --- Lookup Tables [cite: 998-1000] ---
int32_t EXP2_TABLE[256];

// --- Data Structure ---
typedef struct {
    uint8_t *data;
    long length;
} Dataset;

// --- Recurrent State ---
typedef struct {
    int8_t h[N_LAYERS][HIDDEN_DIM];
} RecurrentState;

// --- Model Parameters Struct ---
typedef struct {
    int8_t embedding[VOCAB_SIZE * HIDDEN_DIM];
    int8_t gru_weights[N_LAYERS][4][HIDDEN_DIM * HIDDEN_DIM];
    int8_t gru_biases[N_LAYERS][2][HIDDEN_DIM]; // 0: bf, 1: bh
    int8_t mlp_weights[N_LAYERS][2][HIDDEN_DIM * (HIDDEN_DIM * 4)]; // 0: Expand, 1: Project
    int8_t head[HIDDEN_DIM * VOCAB_SIZE];
    int8_t ln_weights[N_LAYERS][2][HIDDEN_DIM]; // 0: LN1, 1: LN2
    int8_t ln_out[HIDDEN_DIM];
} EggModel;

// --- Helper Functions ---

// Forward Declaration
int32_t compute_loss(int8_t *logits, uint8_t target);

void init_tables() {
    for(int i=0; i<256; i++) 
        EXP2_TABLE[i] = (int32_t)(pow(2.0, (double)i / (1 << FIXED_POINT)) * (1 << FIXED_POINT));
}

int8_t clipped_add(int32_t a, int32_t b) {
    int32_t res = a + b;
    if (res > MAX_VAL) return MAX_VAL;
    if (res < MIN_VAL) return MIN_VAL;
    return (int8_t)res;
}

// Simple RNG helpers for scalar usage
static inline uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static inline int8_t gen_noise_val(uint32_t *rng) {
    uint32_t r = xorshift32(rng);
    return (int8_t)((r & 1 ? 1 : -1) * ((r >> 1) & 31));
}

// Noise generation shared by all backends
static inline void gen_noise_vector(uint32_t *rng, int8_t *out, int count) {
    for (int i = 0; i < count; i += 16) {
        // Generate random bits
        uint32_t r0 = xorshift32(rng);
        uint32_t r1 = xorshift32(rng);
        uint32_t r2 = xorshift32(rng);
        uint32_t r3 = xorshift32(rng);
        
        // We need 16 nibbles + signs. 
        // Actually the original logic: (r&1 ? 1 : -1) * ((r>>1)&15) uses 5 bits per value.
        // We can pack this more efficiently but sticking to xorshift32 is fast enough.
        // Let's scalar generate to a buffer or vectorize the generation logic if bottlenecks.
        // On M4, scalar RNG might be slightly slow compared to vector ops, but matmul is dominant.
        // Let's do a simple block loop.
        for (int j = 0; j < 16; j++) {
             if(i+j < count) out[i+j] = gen_noise_val(rng);
        }
    }
}

#if defined(EGGROLL_USE_AVX2)
static inline int32_t hsum256_epi32(__m256i v) {
    __m128i vlow = _mm256_castsi256_si128(v);
    __m128i vhigh = _mm256_extracti128_si256(v, 1);
    __m128i sum128 = _mm_add_epi32(vlow, vhigh);
    __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    sum128 = _mm_add_epi32(sum128, hi64);
    __m128i hi32 = _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1));
    sum128 = _mm_add_epi32(sum128, hi32);
    return _mm_cvtsi128_si32(sum128);
}
#endif

static inline int32_t dot_product_int8(const int8_t *a, const int8_t *b, int len) {
#if defined(EGGROLL_USE_NEON)
    int32x4_t acc_v = vdupq_n_s32(0);
    int i = 0;
    for (; i <= len - 16; i += 16) {
        int8x16_t va = vld1q_s8(&a[i]);
        int8x16_t vb = vld1q_s8(&b[i]);
        int16x8_t mul_low = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
        int16x8_t mul_high = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
        acc_v = vpadalq_s16(acc_v, mul_low);
        acc_v = vpadalq_s16(acc_v, mul_high);
    }
    int32_t sum = vaddvq_s32(acc_v);
    for (; i < len; i++) sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
#elif defined(EGGROLL_USE_AVX2)
    __m256i acc32 = _mm256_setzero_si256();
    int i = 0;
    for (; i <= len - 32; i += 32) {
        __m128i va_lo = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i va_hi = _mm_loadu_si128((const __m128i*)(a + i + 16));
        __m128i vb_lo = _mm_loadu_si128((const __m128i*)(b + i));
        __m128i vb_hi = _mm_loadu_si128((const __m128i*)(b + i + 16));

        __m256i va16_lo = _mm256_cvtepi8_epi16(va_lo);
        __m256i va16_hi = _mm256_cvtepi8_epi16(va_hi);
        __m256i vb16_lo = _mm256_cvtepi8_epi16(vb_lo);
        __m256i vb16_hi = _mm256_cvtepi8_epi16(vb_hi);

        __m256i prod_lo = _mm256_madd_epi16(va16_lo, vb16_lo);
        __m256i prod_hi = _mm256_madd_epi16(va16_hi, vb16_hi);

        acc32 = _mm256_add_epi32(acc32, prod_lo);
        acc32 = _mm256_add_epi32(acc32, prod_hi);
    }
    int32_t sum = hsum256_epi32(acc32);
    for (; i <= len - 16; i += 16) {
        __m128i va = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i vb = _mm_loadu_si128((const __m128i*)(b + i));
        __m256i va16 = _mm256_cvtepi8_epi16(va);
        __m256i vb16 = _mm256_cvtepi8_epi16(vb);
        __m256i prod = _mm256_madd_epi16(va16, vb16);
        sum += hsum256_epi32(prod);
    }
    for (; i < len; i++) sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
#else
    int32_t sum = 0;
    for (int i = 0; i < len; i++) sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
#endif
}

#if defined(EGGROLL_USE_PTHREADS)
typedef void (*ParallelForFunc)(size_t, void*);

typedef struct {
    pthread_t *threads;
    size_t worker_count;
    size_t hw_threads;
    pthread_mutex_t mutex;
    pthread_cond_t work_cv;
    pthread_cond_t done_cv;
    size_t next_index;
    size_t total;
    size_t running_workers;
    ParallelForFunc func;
    void *ctx;
    int work_available;
    int stop;
    int initialized;
} ThreadPool;

static ThreadPool g_thread_pool = {0};

static void parallel_worker_run(ThreadPool *pool) {
    pthread_mutex_lock(&pool->mutex);
    while (1) {
        while (!pool->work_available && !pool->stop) {
            pthread_cond_wait(&pool->work_cv, &pool->mutex);
        }
        if (pool->stop) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }
        if (pool->next_index >= pool->total) {
            pthread_cond_wait(&pool->done_cv, &pool->mutex);
            continue;
        }
        size_t idx = pool->next_index++;
        pool->running_workers++;
        pthread_mutex_unlock(&pool->mutex);

        pool->func(idx, pool->ctx);

        pthread_mutex_lock(&pool->mutex);
        pool->running_workers--;
        if (pool->next_index >= pool->total && pool->running_workers == 0) {
            pool->work_available = 0;
            pthread_cond_broadcast(&pool->done_cv);
        }
    }
}

static void *parallel_worker_trampoline(void *arg) {
    parallel_worker_run((ThreadPool*)arg);
    return NULL;
}

static void ensure_thread_pool_initialized() {
    if (g_thread_pool.initialized) return;
    long hw_threads = sysconf(_SC_NPROCESSORS_ONLN);
    if (hw_threads < 1) hw_threads = 1;
    g_thread_pool.hw_threads = (size_t)hw_threads;
    g_thread_pool.worker_count = (size_t)hw_threads;
    pthread_mutex_init(&g_thread_pool.mutex, NULL);
    pthread_cond_init(&g_thread_pool.work_cv, NULL);
    pthread_cond_init(&g_thread_pool.done_cv, NULL);
    g_thread_pool.threads = (pthread_t*)malloc(g_thread_pool.worker_count * sizeof(pthread_t));
    for (size_t t = 0; t < g_thread_pool.worker_count; t++) {
        pthread_create(&g_thread_pool.threads[t], NULL, parallel_worker_trampoline, &g_thread_pool);
    }
    g_thread_pool.initialized = 1;
}

static void shutdown_thread_pool() {
    if (!g_thread_pool.initialized) return;
    pthread_mutex_lock(&g_thread_pool.mutex);
    g_thread_pool.stop = 1;
    pthread_cond_broadcast(&g_thread_pool.work_cv);
    pthread_cond_broadcast(&g_thread_pool.done_cv);
    pthread_mutex_unlock(&g_thread_pool.mutex);
    for (size_t t = 0; t < g_thread_pool.worker_count; t++) {
        pthread_join(g_thread_pool.threads[t], NULL);
    }
    free(g_thread_pool.threads);
    pthread_mutex_destroy(&g_thread_pool.mutex);
    pthread_cond_destroy(&g_thread_pool.work_cv);
    pthread_cond_destroy(&g_thread_pool.done_cv);
    memset(&g_thread_pool, 0, sizeof(g_thread_pool));
}

static void run_parallel_for(size_t total, ParallelForFunc func, void *ctx) {
    if (total == 0) return;
    ensure_thread_pool_initialized();
    pthread_mutex_lock(&g_thread_pool.mutex);
    g_thread_pool.total = total;
    g_thread_pool.next_index = 0;
    g_thread_pool.running_workers = 0;
    g_thread_pool.func = func;
    g_thread_pool.ctx = ctx;
    g_thread_pool.work_available = 1;
    pthread_cond_broadcast(&g_thread_pool.work_cv);
    while (g_thread_pool.work_available) {
        pthread_cond_wait(&g_thread_pool.done_cv, &g_thread_pool.mutex);
    }
    pthread_mutex_unlock(&g_thread_pool.mutex);
}
#endif

// --- The Core "Rank-1 Perturbed Matrix Mul" (NEON Optimized) ---
void matmul_perturbed(
    const int8_t *in, const int8_t *w, int8_t *out, 
    int rows, int cols, 
    uint32_t layer_seed, int noise_sign,
    int shift
) {
    // 1. Generate Noise Vectors A and B
    int8_t A[rows];
    int8_t B[cols];
    
    uint32_t rng = layer_seed;
    gen_noise_vector(&rng, A, rows);
    gen_noise_vector(&rng, B, cols);

    int32_t xB = dot_product_int8(in, B, cols);

    // 3. Compute Result
    // xW + noise_sign * (xB * A)
    // We iterate over rows (out_dim), computing dot product of in * W[row]
    
    for(int r=0; r<rows; r++) {
        const int8_t *w_row = &w[r * cols];
        int32_t acc = dot_product_int8(in, w_row, cols);

        if (noise_sign != 0) {
            int32_t noise = (xB * (int32_t)A[r]) * noise_sign;
            acc += (noise >> (FIXED_POINT + SIGMA_SHIFT));
        }
        
        int32_t res = acc >> shift;
        if(res > MAX_VAL) out[r] = MAX_VAL;
        else if(res < MIN_VAL) out[r] = MIN_VAL;
        else out[r] = (int8_t)res;
    }
}

// --- Layer Norm [cite: 965] ---
void egg_ln(const int8_t *x, const int8_t *w, int8_t *out) {
    int32_t sum = 0;
    int i = 0;
#if defined(EGGROLL_USE_NEON)
    int32x4_t sum_v = vdupq_n_s32(0);
    for(; i <= HIDDEN_DIM - 16; i+=16) {
        int8x16_t xv = vld1q_s8(&x[i]);
        int8x16_t abs_xv = vabsq_s8(xv);
        uint16x8_t s1 = vpaddlq_u8(vreinterpretq_u8_s8(abs_xv));
        uint32x4_t s2 = vpaddlq_u16(s1);
        sum_v = vaddq_s32(sum_v, vreinterpretq_s32_u32(s2));
    }
    sum = vaddvq_s32(sum_v);
#elif defined(EGGROLL_USE_AVX2)
    __m256i acc64 = _mm256_setzero_si256();
    const __m256i zero = _mm256_setzero_si256();
    for (; i <= HIDDEN_DIM - 32; i += 32) {
        __m256i xv = _mm256_loadu_si256((const __m256i*)&x[i]);
        __m256i abs_xv = _mm256_abs_epi8(xv);
        __m256i sad = _mm256_sad_epu8(abs_xv, zero);
        acc64 = _mm256_add_epi64(acc64, sad);
    }
    uint64_t tmp[4];
    _mm256_storeu_si256((__m256i*)tmp, acc64);
    sum = (int32_t)(tmp[0] + tmp[1] + tmp[2] + tmp[3]);
#endif
    for(; i<HIDDEN_DIM; i++) sum += abs(x[i]);

    if(sum == 0) sum = 1;
    int32_t mean = sum / HIDDEN_DIM;
    if(mean == 0) mean = 1;
    
    // Vectorize normalization
    for(i=0; i < HIDDEN_DIM; i++) {
        int32_t val = ((int32_t)x[i] * w[i]) / mean;
        if(val > MAX_VAL) out[i] = MAX_VAL;
        else if(val < MIN_VAL) out[i] = MIN_VAL;
        else out[i] = (int8_t)val;
    }
}

// --- Forward Pass (With Noise Injection) [cite: 915] ---
void forward_pass(
    EggModel *model, 
    const uint8_t *inputs, 
    const uint8_t *targets,
    int seq_len, 
    int8_t *logits_out, 
    int32_t *accumulated_loss,
    uint32_t step_seed, 
    int noise_sign,
    RecurrentState *rnn_state
) {
    int8_t x[HIDDEN_DIM], residual[HIDDEN_DIM];
    int8_t buf1[HIDDEN_DIM * 4], buf2[HIDDEN_DIM];
    
    // If no state provided, use a temporary zeroed one (stateless mode)
    RecurrentState local_state;
    if (!rnn_state) {
        memset(&local_state, 0, sizeof(local_state));
        rnn_state = &local_state;
    }
    
    if(accumulated_loss) *accumulated_loss = 0;

    for(int t=0; t<seq_len; t++) {
        // Embedding
        memcpy(x, &model->embedding[inputs[t] * HIDDEN_DIM], HIDDEN_DIM);

        // Layers
        for(int l=0; l<N_LAYERS; l++) {
            uint32_t l_seed = step_seed + (l * 100);

            // -- GRU --
            memcpy(residual, x, HIDDEN_DIM);
            egg_ln(x, model->ln_weights[l][0], x);

            matmul_perturbed(x, model->gru_weights[l][0], buf1, HIDDEN_DIM, HIDDEN_DIM, l_seed+1, noise_sign, 8);
            matmul_perturbed(rnn_state->h[l], model->gru_weights[l][1], buf2, HIDDEN_DIM, HIDDEN_DIM, l_seed+2, noise_sign, 8);
            
            int8_t ft[HIDDEN_DIM];
            for(int i=0; i<HIDDEN_DIM; i++) ft[i] = clipped_add(clipped_add(buf1[i], buf2[i]), model->gru_biases[l][0][i]);

            int8_t gated_past[HIDDEN_DIM];
            for(int i=0; i<HIDDEN_DIM; i++) gated_past[i] = (int8_t)(((int32_t)(ft[i] + 127) * rnn_state->h[l][i]) >> 8);

            matmul_perturbed(x, model->gru_weights[l][2], buf1, HIDDEN_DIM, HIDDEN_DIM, l_seed+3, noise_sign, 8);
            matmul_perturbed(gated_past, model->gru_weights[l][3], buf2, HIDDEN_DIM, HIDDEN_DIM, l_seed+4, noise_sign, 8);
            
            int8_t ht[HIDDEN_DIM];
            for(int i=0; i<HIDDEN_DIM; i++) ht[i] = clipped_add(clipped_add(buf1[i], buf2[i]), model->gru_biases[l][1][i]);

            // State Update
            for(int i=0; i<HIDDEN_DIM; i++) {
                int32_t update = ((int32_t)(ft[i] + 127) * (ht[i] - rnn_state->h[l][i])) >> 8;
                rnn_state->h[l][i] = clipped_add(rnn_state->h[l][i], update);
                x[i] = rnn_state->h[l][i]; // Output is new state
            }
            
            // Residual Add
            for(int i=0; i<HIDDEN_DIM; i++) x[i] = clipped_add(x[i], residual[i]);

            // -- MLP --
            memcpy(residual, x, HIDDEN_DIM);
            egg_ln(x, model->ln_weights[l][1], x);
            
            // Expand: Hidden -> 4*Hidden
            // Cols = HIDDEN_DIM (256). Sqrt(256)=16. Shift = 4+4=8.
            matmul_perturbed(x, model->mlp_weights[l][0], buf1, HIDDEN_DIM * 4, HIDDEN_DIM, l_seed+5, noise_sign, 8);
            
            // Project: 4*Hidden -> Hidden
            // Cols = HIDDEN_DIM*4 (1024). Sqrt(1024)=32. Shift = 4+5=9.
            matmul_perturbed(buf1, model->mlp_weights[l][1], x, HIDDEN_DIM, HIDDEN_DIM * 4, l_seed+6, noise_sign, 9);

            for(int i=0; i<HIDDEN_DIM; i++) x[i] = clipped_add(x[i], residual[i]);
        }

        // Final Head
        egg_ln(x, model->ln_out, x);
        matmul_perturbed(x, model->head, logits_out, VOCAB_SIZE, HIDDEN_DIM, step_seed+999, noise_sign, 8);
        
        if(targets && accumulated_loss) {
            *accumulated_loss += compute_loss(logits_out, targets[t]);
        }
    }
}

// --- Helper for Loss Calculation ---
static inline int get_msb(uint32_t n) {
    int pos = 0;
    if (n >= 1<<16) { n >>= 16; pos += 16; }
    if (n >= 1<<8)  { n >>= 8;  pos += 8; }
    if (n >= 1<<4)  { n >>= 4;  pos += 4; }
    if (n >= 1<<2)  { n >>= 2;  pos += 2; }
    if (n >= 1<<1)  {           pos += 1; }
    return pos;
}

int32_t log2_fixed(int32_t x) {
    if (x <= 0) return 0;
    int k = get_msb(x);
    int32_t fraction;
    if (k >= 4) {
        fraction = (x - (1 << k)) >> (k - 4);
    } else {
        fraction = (x - (1 << k)) << (4 - k);
    }
    return (k << 4) + fraction - 64;
}

int32_t compute_loss(int8_t *logits, uint8_t target) {
    int32_t sum_exp = 0;
    for(int i=0; i<VOCAB_SIZE; i++) {
        int idx = (int32_t)logits[i] + 128;
        sum_exp += EXP2_TABLE[idx < 0 ? 0 : (idx > 255 ? 255 : idx)];
    }
    int32_t log_sum = log2_fixed(sum_exp);
    int32_t target_logit = (int32_t)logits[target] + 128;
    return log_sum - target_logit;
}

void update_matrix(
    int8_t *W, int rows, int cols, 
    uint32_t seed, 
    const int *fitnesses, 
    int pop_size
) {
    // Optimized "Inverted Loop" Strategy:
    // 1. Pre-compute and transpose noise vectors into compact buffers.
    // 2. Single pass over weights to compute votes (dot product over population) and update.
    // This eliminates the large 'votes' memset and repeated sweeps.

    // Transposed buffers: [index][population_pair_index]
    // A_T stores A * f directly.
    static int8_t A_T[MAX_MATRIX_DIM][MAX_POP_PAIRS] __attribute__((aligned(16)));
    static int8_t B_T[MAX_MATRIX_DIM][MAX_POP_PAIRS] __attribute__((aligned(16)));
    
    int pairs = pop_size / 2;
    if (pairs > MAX_POP_PAIRS) pairs = MAX_POP_PAIRS;

    // 1. Pre-compute Noise and Transpose
    for(int p=0; p<pairs; p++) {
        int f = fitnesses[p];
        uint32_t rng = seed + p;

        int8_t A_temp[MAX_MATRIX_DIM];
        int8_t B_temp[MAX_MATRIX_DIM];
        
        gen_noise_vector(&rng, A_temp, rows);
        gen_noise_vector(&rng, B_temp, cols);

        if (f == 0) {
            for(int r=0; r<rows; r++) A_T[r][p] = 0;
        } else {
            for(int r=0; r<rows; r++) A_T[r][p] = (int8_t)(A_temp[r] * f);
            for(int c=0; c<cols; c++) B_T[c][p] = B_temp[c];
        }
    }

    // 2. Compute Votes and Update Weights (Single Pass)
    for(int r=0; r<rows; r++) {
        int8_t *w_row = &W[r * cols];
        int8_t *a_ptr = A_T[r];

        for(int c=0; c<cols; c++) {
            int8_t *b_ptr = B_T[c];
            int32_t vote = dot_product_int8(a_ptr, b_ptr, pairs);

            // Apply Update
            if(vote > UPDATE_THRESHOLD && w_row[c] < MAX_VAL) w_row[c]++;
            else if(vote < -UPDATE_THRESHOLD && w_row[c] > MIN_VAL) w_row[c]--;
        }
    }
}

typedef struct {
    EggModel *model;
    Dataset *ds;
    long start_idx;
    uint32_t step_seed;
    RecurrentState *pop_states;
    int *pair_fitnesses;
    long stride;
    long ds_len_minus_seq;
} PopulationEvalContext;

static void process_population_pair(size_t p_idx, void *ctx_void) {
    PopulationEvalContext *ctx = (PopulationEvalContext*)ctx_void;
    uint32_t p_seed = ctx->step_seed + (uint32_t)p_idx;
    int8_t local_logits[VOCAB_SIZE];
    int32_t loss_pos = 0, loss_neg = 0;

    long stream_idx = (ctx->start_idx + (p_idx * ctx->stride)) % ctx->ds_len_minus_seq;

    forward_pass(ctx->model, &ctx->ds->data[stream_idx], &ctx->ds->data[stream_idx+1], SEQ_LEN, local_logits, &loss_pos, p_seed, 1, &ctx->pop_states[p_idx*2]);
    forward_pass(ctx->model, &ctx->ds->data[stream_idx], &ctx->ds->data[stream_idx+1], SEQ_LEN, local_logits, &loss_neg, p_seed, -1, &ctx->pop_states[p_idx*2+1]);

    if (loss_pos < loss_neg) ctx->pair_fitnesses[p_idx] = 1;
    else if (loss_neg < loss_pos) ctx->pair_fitnesses[p_idx] = -1;
    else ctx->pair_fitnesses[p_idx] = 0;
}

// --- Sampling Helper ---
#define COLOR_GREEN "\033[32m"
#define COLOR_CYAN  "\033[36m"
#define COLOR_RESET "\033[0m"

int sample_logits(int8_t *logits) {
    int32_t probs[VOCAB_SIZE];
    int32_t sum = 0;
    for(int i=0; i<VOCAB_SIZE; i++) {
        int idx = (int32_t)logits[i] + 128;
        idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
        probs[i] = EXP2_TABLE[idx];
        sum += probs[i];
    }
    if(sum == 0) return 0;
    int32_t r = rand() % sum;
    int32_t acc = 0;
    for(int i=0; i<VOCAB_SIZE; i++) {
        acc += probs[i];
        if(r < acc) return i;
    }
    return VOCAB_SIZE - 1;
}

void sample_model(EggModel *model, const uint8_t *seed_text, int seed_len, int gen_len) {
    int8_t logits[VOCAB_SIZE];
    RecurrentState state;
    memset(&state, 0, sizeof(state));

    printf(COLOR_GREEN);
    int input_token = 0;
    
    for(int t=0; t < seed_len + gen_len; t++) {
        if (t < seed_len) {
            input_token = seed_text[t];
            if(input_token >= 32 && input_token <= 126) printf("%c", input_token);
            else printf(".");
        } else {
            if (t == seed_len) printf(COLOR_CYAN);
            if(input_token >= 32 && input_token <= 126) printf("%c", input_token);
            else printf(".");
        }

        // Infer Only - No Noise
        forward_pass(model, (uint8_t*)&input_token, NULL, 1, logits, NULL, 0, 0, &state);
        // Note: forward_pass handles state update internally for seq_len=1
        
        if (t >= seed_len - 1) {
            input_token = sample_logits(logits);
        }
    }
    printf(COLOR_RESET "\n");
}

Dataset load_data(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if(!f) { printf("Error: Create 'input.txt' first.\n"); exit(1); }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *data = (uint8_t*)malloc(len);
    fread(data, 1, len, f);
    fclose(f);
    return (Dataset){data, len};
}

int main() {
    srand(time(NULL));
    init_tables();
    Dataset ds = load_data("input.txt");
    printf("Loaded dataset: %ld bytes\n", ds.length);
    printf("CPU backend: %s | Parallel: %s\n", EGGROLL_SIMD_NAME, EGGROLL_PARALLEL_BACKEND);

    EggModel *model;
    posix_memalign((void**)&model, 16, sizeof(EggModel));
    memset(model, 0, sizeof(EggModel));

    uint32_t init_rng = 42;
    for(int i=0; i<VOCAB_SIZE*HIDDEN_DIM; i++) model->embedding[i] = gen_noise_val(&init_rng);
    for(int i=0; i<HIDDEN_DIM*VOCAB_SIZE; i++) model->head[i] = gen_noise_val(&init_rng);
    
    for(int l=0; l<N_LAYERS; l++) {
        for(int g=0; g<4; g++) {
            for(int i=0; i<HIDDEN_DIM*HIDDEN_DIM; i++) {
                model->gru_weights[l][g][i] = gen_noise_val(&init_rng);
            }
        }
        for(int m=0; m<2; m++) {
             // Utilize the full allocated space: HIDDEN * (HIDDEN * 4)
             for(int i=0; i<HIDDEN_DIM*(HIDDEN_DIM*4); i++) {
                 model->mlp_weights[l][m][i] = gen_noise_val(&init_rng);
             }
        }
    }

    for(int l=0; l<N_LAYERS; l++) {
        for(int i=0; i<HIDDEN_DIM; i++) model->ln_weights[l][0][i] = 16;
        for(int i=0; i<HIDDEN_DIM; i++) model->ln_weights[l][1][i] = 16;
        for(int i=0; i<HIDDEN_DIM; i++) model->ln_out[i] = 16;
    }

    int *pair_fitnesses = (int*)malloc((POPULATION_SIZE/2) * sizeof(int));
    int8_t *logits = (int8_t*)malloc(VOCAB_SIZE);
    
    // Persistent States
    RecurrentState *pop_states = (RecurrentState*)aligned_alloc(16, POPULATION_SIZE * sizeof(RecurrentState));
    RecurrentState main_state;
    memset(pop_states, 0, POPULATION_SIZE * sizeof(RecurrentState));
    memset(&main_state, 0, sizeof(RecurrentState));

    printf("Starting EGGROLL Training (Stateful + Optimized)...\n");
    
    struct timespec start_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    long total_tokens = 0;
    long max_steps = (ds.length - 1) / SEQ_LEN;

    for(long step=0; step < max_steps; step++) {
        // Use a more robust seed mixing to avoid correlations if time(NULL) doesn't change
        uint32_t step_seed = (uint32_t)time(NULL) ^ (step * 0x9e3779b9);
        int start_idx = step * SEQ_LEN;
        
        if(step % 10 == 0) {
            sample_model(model, &ds.data[start_idx], 30, 30);
            
            int32_t loss_val = 0;
            forward_pass(model, &ds.data[start_idx], &ds.data[start_idx+1], SEQ_LEN, logits, &loss_val, step_seed, 0, &main_state);
            
            struct timespec current_time;
            clock_gettime(CLOCK_MONOTONIC, &current_time);
            double elapsed_sec = (current_time.tv_sec - start_time.tv_sec) + 
                                 (current_time.tv_nsec - start_time.tv_nsec) / 1e9;
            double tps = (elapsed_sec > 0) ? (double)total_tokens / elapsed_sec : 0.0;
            // Loss is accumulated fixed point. Divide by SEQ_LEN to get average per token, then by 2^FIXED_POINT
            printf("Step %ld/%ld | Loss: %.4f | Tok/s: %.2f\n", step, max_steps, (double)loss_val / (SEQ_LEN * (1 << FIXED_POINT)), tps);
        }

        size_t pop_pairs = POPULATION_SIZE / 2;
        if (pop_pairs == 0) pop_pairs = 1;
        PopulationEvalContext eval_ctx = {
            .model = model,
            .ds = &ds,
            .start_idx = start_idx,
            .step_seed = step_seed,
            .pop_states = pop_states,
            .pair_fitnesses = pair_fitnesses,
            .stride = (long)(ds.length / pop_pairs),
            .ds_len_minus_seq = (ds.length > SEQ_LEN) ? (ds.length - SEQ_LEN) : 1
        };
        if (eval_ctx.stride <= 0) eval_ctx.stride = 1;
#if defined(__APPLE__)
        dispatch_apply(pop_pairs, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t p_idx) {
            process_population_pair(p_idx, &eval_ctx);
        });
#elif defined(EGGROLL_USE_PTHREADS)
        run_parallel_for(pop_pairs, process_population_pair, &eval_ctx);
#else
        for (size_t p_idx = 0; p_idx < pop_pairs; ++p_idx) {
            process_population_pair(p_idx, &eval_ctx);
        }
#endif

        for(int l=0; l<N_LAYERS; l++) {
            uint32_t l_seed = step_seed + (l * 100);
            update_matrix(model->gru_weights[l][0], HIDDEN_DIM, HIDDEN_DIM, l_seed+1, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->gru_weights[l][1], HIDDEN_DIM, HIDDEN_DIM, l_seed+2, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->gru_weights[l][2], HIDDEN_DIM, HIDDEN_DIM, l_seed+3, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->gru_weights[l][3], HIDDEN_DIM, HIDDEN_DIM, l_seed+4, pair_fitnesses, POPULATION_SIZE);
            
            update_matrix(model->mlp_weights[l][0], HIDDEN_DIM * 4, HIDDEN_DIM, l_seed+5, pair_fitnesses, POPULATION_SIZE);
            update_matrix(model->mlp_weights[l][1], HIDDEN_DIM, HIDDEN_DIM * 4, l_seed+6, pair_fitnesses, POPULATION_SIZE);
        }
        update_matrix(model->head, VOCAB_SIZE, HIDDEN_DIM, step_seed+999, pair_fitnesses, POPULATION_SIZE);

        total_tokens += SEQ_LEN;
    }

    printf("Training Done.\n");
    free(ds.data); free(model); free(logits); free(pair_fitnesses);
    free(pop_states);
#if defined(EGGROLL_USE_PTHREADS)
    shutdown_thread_pool();
#endif
    return 0;
}
