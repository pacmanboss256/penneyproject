# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_GET_SIZE
from libc.stdint cimport uint32_t, uint8_t, uint64_t
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
import numpy as np
cimport numpy as cnp


cdef extern from *:
    """
    #include <Python.h>
    #include <immintrin.h>
    #include <stdint.h>

    #if defined(__GNUC__)
    #define TARGET_AVX2 __attribute__((target("avx2")))
    #define TARGET_AVX512 __attribute__((target("avx512bw")))
    #else
    #define TARGET_AVX2
    #define TARGET_AVX512
    #endif

    static inline int cpu_supports_avx512(void) {
    #if defined(__x86_64__) || defined(__i386__)
        #if defined(__GNUC__) || defined(__clang__)
            return __builtin_cpu_supports("avx512bw");
        #else
            return 0;
        #endif
    #else
        return 0;
    #endif
    }

    static inline int cpu_supports_avx2(void) {
    #if defined(__x86_64__) || defined(__i386__)
        #if defined(__GNUC__) || defined(__clang__)
            return __builtin_cpu_supports("avx2");
        #else
            return 0;
        #endif
    #else
        return 0;
    #endif
    }

    static const uint32_t STEP3_MASKS_32[3] = {0x49249249u, 0x24924924u, 0x92492492u};
    static const uint32_t STEP4_MASKS_32[4] = {0x11111111u, 0x88888888u, 0x44444444u, 0x22222222u};

    static const uint64_t STEP3_MASKS_64[3] = {0x9249249249249249ull, 0x4924924924924924ull, 0x2492492492492492ull};
    static const uint64_t STEP4_MASKS_64[4] = {0x1111111111111111ull, 0x8888888888888888ull, 0x4444444444444444ull, 0x2222222222222222ull};

    static inline uint32_t step_mask32(int step, int mod) {
        if (step == 3) {
            return STEP3_MASKS_32[mod];
        } else if (step == 4) {
            return STEP4_MASKS_32[mod];
        }
        return 0xffffffffu;
    }

    static inline uint64_t step_mask64(int step, int mod) {
        if (step == 3) {
            return STEP3_MASKS_64[mod];
        } else if (step == 4) {
            return STEP4_MASKS_64[mod];
        }
        return 0xffffffffffffffffull;
    }

    TARGET_AVX2 static inline int find_match3_avx2(
        const uint8_t* s,
        Py_ssize_t n,
        Py_ssize_t start,
        Py_ssize_t step,
        uint32_t p1t,
        uint32_t p2t,
        Py_ssize_t* out_pos,
        int* out_pat
    ) {
        Py_ssize_t limit = n - 3;
        if (limit < start) {
            return 0;
        }

        const uint8_t p1b0 = (uint8_t)(p1t >> 16);
        const uint8_t p1b1 = (uint8_t)(p1t >> 8);
        const uint8_t p1b2 = (uint8_t)(p1t);
        const uint8_t p2b0 = (uint8_t)(p2t >> 16);
        const uint8_t p2b1 = (uint8_t)(p2t >> 8);
        const uint8_t p2b2 = (uint8_t)(p2t);

        const __m256i p1v0 = _mm256_set1_epi8((char)p1b0);
        const __m256i p1v1 = _mm256_set1_epi8((char)p1b1);
        const __m256i p1v2 = _mm256_set1_epi8((char)p1b2);
        const __m256i p2v0 = _mm256_set1_epi8((char)p2b0);
        const __m256i p2v1 = _mm256_set1_epi8((char)p2b1);
        const __m256i p2v2 = _mm256_set1_epi8((char)p2b2);

        Py_ssize_t i = start;
        const Py_ssize_t vec_end = limit - 31;
        int mod = 0;
        if (step != 1) {
            mod = (int)((i - start) % step);
        }

        for (; i <= vec_end; i += 32) {
            const __m256i v0 = _mm256_loadu_si256((const __m256i*)(s + i));
            const __m256i v1 = _mm256_loadu_si256((const __m256i*)(s + i + 1));
            const __m256i v2 = _mm256_loadu_si256((const __m256i*)(s + i + 2));

            __m256i c10 = _mm256_cmpeq_epi8(v0, p1v0);
            __m256i c11 = _mm256_cmpeq_epi8(v1, p1v1);
            __m256i c12 = _mm256_cmpeq_epi8(v2, p1v2);
            __m256i m1v = _mm256_and_si256(_mm256_and_si256(c10, c11), c12);
            uint32_t m1 = (uint32_t)_mm256_movemask_epi8(m1v);

            __m256i c20 = _mm256_cmpeq_epi8(v0, p2v0);
            __m256i c21 = _mm256_cmpeq_epi8(v1, p2v1);
            __m256i c22 = _mm256_cmpeq_epi8(v2, p2v2);
            __m256i m2v = _mm256_and_si256(_mm256_and_si256(c20, c21), c22);
            uint32_t m2 = (uint32_t)_mm256_movemask_epi8(m2v);

            if (step != 1) {
                const uint32_t allow = step_mask32((int)step, mod);
                m1 &= allow;
                m2 &= allow;
            }

            uint32_t m = m1 | m2;
            if (m) {
                const int bit = __builtin_ctz(m);
                *out_pos = i + bit;
                *out_pat = (m1 & (1u << bit)) ? 1 : 2;
                return 1;
            }

            if (step != 1) {
                mod += (32 % step);
                if (mod >= step) {
                    mod -= step;
                }
            }
        }

        if (step != 1) {
            Py_ssize_t rem = (i - start) % step;
            if (rem) {
                i += step - rem;
            }
        }

        for (; i <= limit; i += step) {
            uint32_t tri = ((uint32_t)s[i] << 16) | ((uint32_t)s[i + 1] << 8) | (uint32_t)s[i + 2];
            if (tri == p1t) {
                *out_pos = i;
                *out_pat = 1;
                return 1;
            }
            if (tri == p2t) {
                *out_pos = i;
                *out_pat = 2;
                return 1;
            }
        }

        return 0;
    }

    TARGET_AVX2 static inline int find_match4_avx2(
        const uint8_t* s,
        Py_ssize_t n,
        Py_ssize_t start,
        Py_ssize_t step,
        uint32_t p1t,
        uint32_t p2t,
        Py_ssize_t* out_pos,
        int* out_pat
    ) {
        Py_ssize_t limit = n - 4;
        if (limit < start) {
            return 0;
        }

        const uint8_t p1b0 = (uint8_t)(p1t >> 24);
        const uint8_t p1b1 = (uint8_t)(p1t >> 16);
        const uint8_t p1b2 = (uint8_t)(p1t >> 8);
        const uint8_t p1b3 = (uint8_t)(p1t);
        const uint8_t p2b0 = (uint8_t)(p2t >> 24);
        const uint8_t p2b1 = (uint8_t)(p2t >> 16);
        const uint8_t p2b2 = (uint8_t)(p2t >> 8);
        const uint8_t p2b3 = (uint8_t)(p2t);

        const __m256i p1v0 = _mm256_set1_epi8((char)p1b0);
        const __m256i p1v1 = _mm256_set1_epi8((char)p1b1);
        const __m256i p1v2 = _mm256_set1_epi8((char)p1b2);
        const __m256i p1v3 = _mm256_set1_epi8((char)p1b3);
        const __m256i p2v0 = _mm256_set1_epi8((char)p2b0);
        const __m256i p2v1 = _mm256_set1_epi8((char)p2b1);
        const __m256i p2v2 = _mm256_set1_epi8((char)p2b2);
        const __m256i p2v3 = _mm256_set1_epi8((char)p2b3);

        Py_ssize_t i = start;
        const Py_ssize_t vec_end = limit - 31;
        int mod = 0;
        if (step != 1) {
            mod = (int)((i - start) % step);
        }

        for (; i <= vec_end; i += 32) {
            const __m256i v0 = _mm256_loadu_si256((const __m256i*)(s + i));
            const __m256i v1 = _mm256_loadu_si256((const __m256i*)(s + i + 1));
            const __m256i v2 = _mm256_loadu_si256((const __m256i*)(s + i + 2));
            const __m256i v3 = _mm256_loadu_si256((const __m256i*)(s + i + 3));

            __m256i c10 = _mm256_cmpeq_epi8(v0, p1v0);
            __m256i c11 = _mm256_cmpeq_epi8(v1, p1v1);
            __m256i c12 = _mm256_cmpeq_epi8(v2, p1v2);
            __m256i c13 = _mm256_cmpeq_epi8(v3, p1v3);
            __m256i m1v = _mm256_and_si256(_mm256_and_si256(c10, c11), _mm256_and_si256(c12, c13));
            uint32_t m1 = (uint32_t)_mm256_movemask_epi8(m1v);

            __m256i c20 = _mm256_cmpeq_epi8(v0, p2v0);
            __m256i c21 = _mm256_cmpeq_epi8(v1, p2v1);
            __m256i c22 = _mm256_cmpeq_epi8(v2, p2v2);
            __m256i c23 = _mm256_cmpeq_epi8(v3, p2v3);
            __m256i m2v = _mm256_and_si256(_mm256_and_si256(c20, c21), _mm256_and_si256(c22, c23));
            uint32_t m2 = (uint32_t)_mm256_movemask_epi8(m2v);

            if (step != 1) {
                const uint32_t allow = step_mask32((int)step, mod);
                m1 &= allow;
                m2 &= allow;
            }

            uint32_t m = m1 | m2;
            if (m) {
                const int bit = __builtin_ctz(m);
                *out_pos = i + bit;
                *out_pat = (m1 & (1u << bit)) ? 1 : 2;
                return 1;
            }

            if (step != 1) {
                mod += (32 % step);
                if (mod >= step) {
                    mod -= step;
                }
            }
        }

        if (step != 1) {
            Py_ssize_t rem = (i - start) % step;
            if (rem) {
                i += step - rem;
            }
        }

        for (; i <= limit; i += step) {
            uint32_t quad = ((uint32_t)s[i] << 24) | ((uint32_t)s[i + 1] << 16) | ((uint32_t)s[i + 2] << 8) | (uint32_t)s[i + 3];
            if (quad == p1t) {
                *out_pos = i;
                *out_pat = 1;
                return 1;
            }
            if (quad == p2t) {
                *out_pos = i;
                *out_pat = 2;
                return 1;
            }
        }

        return 0;
    }

    TARGET_AVX512 static inline int find_match3_avx512(
        const uint8_t* s,
        Py_ssize_t n,
        Py_ssize_t start,
        Py_ssize_t step,
        uint32_t p1t,
        uint32_t p2t,
        Py_ssize_t* out_pos,
        int* out_pat
    ) {
        Py_ssize_t limit = n - 3;
        if (limit < start) {
            return 0;
        }

        const uint8_t p1b0 = (uint8_t)(p1t >> 16);
        const uint8_t p1b1 = (uint8_t)(p1t >> 8);
        const uint8_t p1b2 = (uint8_t)(p1t);
        const uint8_t p2b0 = (uint8_t)(p2t >> 16);
        const uint8_t p2b1 = (uint8_t)(p2t >> 8);
        const uint8_t p2b2 = (uint8_t)(p2t);

        const __m512i p1v0 = _mm512_set1_epi8((char)p1b0);
        const __m512i p1v1 = _mm512_set1_epi8((char)p1b1);
        const __m512i p1v2 = _mm512_set1_epi8((char)p1b2);
        const __m512i p2v0 = _mm512_set1_epi8((char)p2b0);
        const __m512i p2v1 = _mm512_set1_epi8((char)p2b1);
        const __m512i p2v2 = _mm512_set1_epi8((char)p2b2);

        Py_ssize_t i = start;
        const Py_ssize_t vec_end = limit - 63;
        int mod = 0;
        if (step != 1) {
            mod = (int)((i - start) % step);
        }

        for (; i <= vec_end; i += 64) {
            const __m512i v0 = _mm512_loadu_si512((const void*)(s + i));
            const __m512i v1 = _mm512_loadu_si512((const void*)(s + i + 1));
            const __m512i v2 = _mm512_loadu_si512((const void*)(s + i + 2));

            __mmask64 m1 = _mm512_cmpeq_epi8_mask(v0, p1v0);
            m1 &= _mm512_cmpeq_epi8_mask(v1, p1v1);
            m1 &= _mm512_cmpeq_epi8_mask(v2, p1v2);

            __mmask64 m2 = _mm512_cmpeq_epi8_mask(v0, p2v0);
            m2 &= _mm512_cmpeq_epi8_mask(v1, p2v1);
            m2 &= _mm512_cmpeq_epi8_mask(v2, p2v2);

            if (step != 1) {
                const uint64_t allow = step_mask64((int)step, mod);
                m1 &= allow;
                m2 &= allow;
            }

            __mmask64 m = m1 | m2;
            if (m) {
                const int bit = __builtin_ctzll((unsigned long long)m);
                *out_pos = i + bit;
                *out_pat = (m1 & (1ull << bit)) ? 1 : 2;
                return 1;
            }

            if (step != 1) {
                mod += (64 % step);
                if (mod >= step) {
                    mod -= step;
                }
            }
        }

        if (step != 1) {
            Py_ssize_t rem = (i - start) % step;
            if (rem) {
                i += step - rem;
            }
        }

        for (; i <= limit; i += step) {
            uint32_t tri = ((uint32_t)s[i] << 16) | ((uint32_t)s[i + 1] << 8) | (uint32_t)s[i + 2];
            if (tri == p1t) {
                *out_pos = i;
                *out_pat = 1;
                return 1;
            }
            if (tri == p2t) {
                *out_pos = i;
                *out_pat = 2;
                return 1;
            }
        }

        return 0;
    }

    TARGET_AVX512 static inline int find_match4_avx512(
        const uint8_t* s,
        Py_ssize_t n,
        Py_ssize_t start,
        Py_ssize_t step,
        uint32_t p1t,
        uint32_t p2t,
        Py_ssize_t* out_pos,
        int* out_pat
    ) {
        Py_ssize_t limit = n - 4;
        if (limit < start) {
            return 0;
        }

        const uint8_t p1b0 = (uint8_t)(p1t >> 24);
        const uint8_t p1b1 = (uint8_t)(p1t >> 16);
        const uint8_t p1b2 = (uint8_t)(p1t >> 8);
        const uint8_t p1b3 = (uint8_t)(p1t);
        const uint8_t p2b0 = (uint8_t)(p2t >> 24);
        const uint8_t p2b1 = (uint8_t)(p2t >> 16);
        const uint8_t p2b2 = (uint8_t)(p2t >> 8);
        const uint8_t p2b3 = (uint8_t)(p2t);

        const __m512i p1v0 = _mm512_set1_epi8((char)p1b0);
        const __m512i p1v1 = _mm512_set1_epi8((char)p1b1);
        const __m512i p1v2 = _mm512_set1_epi8((char)p1b2);
        const __m512i p1v3 = _mm512_set1_epi8((char)p1b3);
        const __m512i p2v0 = _mm512_set1_epi8((char)p2b0);
        const __m512i p2v1 = _mm512_set1_epi8((char)p2b1);
        const __m512i p2v2 = _mm512_set1_epi8((char)p2b2);
        const __m512i p2v3 = _mm512_set1_epi8((char)p2b3);

        Py_ssize_t i = start;
        const Py_ssize_t vec_end = limit - 63;
        int mod = 0;
        if (step != 1) {
            mod = (int)((i - start) % step);
        }

        for (; i <= vec_end; i += 64) {
            const __m512i v0 = _mm512_loadu_si512((const void*)(s + i));
            const __m512i v1 = _mm512_loadu_si512((const void*)(s + i + 1));
            const __m512i v2 = _mm512_loadu_si512((const void*)(s + i + 2));
            const __m512i v3 = _mm512_loadu_si512((const void*)(s + i + 3));

            __mmask64 m1 = _mm512_cmpeq_epi8_mask(v0, p1v0);
            m1 &= _mm512_cmpeq_epi8_mask(v1, p1v1);
            m1 &= _mm512_cmpeq_epi8_mask(v2, p1v2);
            m1 &= _mm512_cmpeq_epi8_mask(v3, p1v3);

            __mmask64 m2 = _mm512_cmpeq_epi8_mask(v0, p2v0);
            m2 &= _mm512_cmpeq_epi8_mask(v1, p2v1);
            m2 &= _mm512_cmpeq_epi8_mask(v2, p2v2);
            m2 &= _mm512_cmpeq_epi8_mask(v3, p2v3);

            if (step != 1) {
                const uint64_t allow = step_mask64((int)step, mod);
                m1 &= allow;
                m2 &= allow;
            }

            __mmask64 m = m1 | m2;
            if (m) {
                const int bit = __builtin_ctzll((unsigned long long)m);
                *out_pos = i + bit;
                *out_pat = (m1 & (1ull << bit)) ? 1 : 2;
                return 1;
            }

            if (step != 1) {
                mod += (64 % step);
                if (mod >= step) {
                    mod -= step;
                }
            }
        }

        if (step != 1) {
            Py_ssize_t rem = (i - start) % step;
            if (rem) {
                i += step - rem;
            }
        }

        for (; i <= limit; i += step) {
            uint32_t quad = ((uint32_t)s[i] << 24) | ((uint32_t)s[i + 1] << 16) | ((uint32_t)s[i + 2] << 8) | (uint32_t)s[i + 3];
            if (quad == p1t) {
                *out_pos = i;
                *out_pat = 1;
                return 1;
            }
            if (quad == p2t) {
                *out_pos = i;
                *out_pat = 2;
                return 1;
            }
        }

        return 0;
    }
    """
    int cpu_supports_avx512() nogil
    int cpu_supports_avx2() nogil

    int find_match3_avx2(const uint8_t* s, Py_ssize_t n, Py_ssize_t start, Py_ssize_t step,
                         uint32_t p1t, uint32_t p2t, Py_ssize_t* out_pos, int* out_pat) nogil
    int find_match4_avx2(const uint8_t* s, Py_ssize_t n, Py_ssize_t start, Py_ssize_t step,
                         uint32_t p1t, uint32_t p2t, Py_ssize_t* out_pos, int* out_pat) nogil
    int find_match3_avx512(const uint8_t* s, Py_ssize_t n, Py_ssize_t start, Py_ssize_t step,
                           uint32_t p1t, uint32_t p2t, Py_ssize_t* out_pos, int* out_pat) nogil
    int find_match4_avx512(const uint8_t* s, Py_ssize_t n, Py_ssize_t start, Py_ssize_t step,
                           uint32_t p1t, uint32_t p2t, Py_ssize_t* out_pos, int* out_pat) nogil


cdef int _simd_level = -1
cdef int _simd_printed = 0


cdef inline void _init_simd() noexcept:
    global _simd_level
    if _simd_level < 0:
        if cpu_supports_avx512():
            _simd_level = 2
        elif cpu_supports_avx2():
            _simd_level = 1
        else:
            _simd_level = 0


cdef inline uint32_t pack3(const uint8_t* s, Py_ssize_t i) noexcept nogil:
    # pack 3 bytes into a single uint32
    return ((<uint32_t>s[i]   << 16) |
            (<uint32_t>s[i+1] << 8)  |
            (<uint32_t>s[i+2]))


cdef inline uint32_t pack4(const uint8_t* s, Py_ssize_t i) noexcept nogil:
    # pack 4 bytes into a single uint32
    return ((<uint32_t>s[i]   << 24) |
            (<uint32_t>s[i+1] << 16) |
            (<uint32_t>s[i+2] << 8)  |
            (<uint32_t>s[i+3]))


cdef inline void score_one3(const uint8_t* s, Py_ssize_t n,
                            uint32_t p1t, uint32_t p2t,
                            bint aligned,
                            long* p1score, long* p2score, long* draw) noexcept nogil:
    cdef Py_ssize_t offset = 0
    cdef Py_ssize_t i
    cdef Py_ssize_t step = 3 if aligned else 1
    cdef uint32_t tri
    cdef bint found
    cdef int which

    p1score[0] = 0
    p2score[0] = 0
    draw[0] = 0
    if n < 3:
        draw[0] = 1
        return

    while offset <= n - 3:
        found = False
        if _simd_level == 2:
            if find_match3_avx512(s, n, offset, step, p1t, p2t, &i, &which):
                found = True
        elif _simd_level == 1:
            if find_match3_avx2(s, n, offset, step, p1t, p2t, &i, &which):
                found = True
        else:
            i = offset
            while i <= n - 3:
                tri = pack3(s, i)
                if tri == p1t:
                    which = 1
                    found = True
                    break
                elif tri == p2t:
                    which = 2
                    found = True
                    break
                i += step
        if not found:
            draw[0] += 1
            break
        if which == 1:
            p1score[0] += (i - offset) + 3
            offset = i + 3
        else:
            p2score[0] += (i - offset) + 3
            offset = i + 3

    if p1score[0] == p2score[0]:
        draw[0] += p1score[0] + p2score[0] + 1


cdef inline void score_one4(const uint8_t* s, Py_ssize_t n,
                            uint32_t p1t, uint32_t p2t,
                            bint aligned,
                            long* p1score, long* p2score, long* draw) noexcept nogil:
    cdef Py_ssize_t offset = 0
    cdef Py_ssize_t i
    cdef Py_ssize_t step = 4 if aligned else 1
    cdef uint32_t quad
    cdef bint found
    cdef int which

    p1score[0] = 0
    p2score[0] = 0
    draw[0] = 0
    if n < 4:
        draw[0] = 1
        return

    while offset <= n - 4:
        found = False
        if _simd_level == 2:
            if find_match4_avx512(s, n, offset, step, p1t, p2t, &i, &which):
                found = True
        elif _simd_level == 1:
            if find_match4_avx2(s, n, offset, step, p1t, p2t, &i, &which):
                found = True
        else:
            i = offset
            while i <= n - 4:
                quad = pack4(s, i)
                if quad == p1t:
                    which = 1
                    found = True
                    break
                elif quad == p2t:
                    which = 2
                    found = True
                    break
                i += step
        if not found:
            draw[0] += 1
            break
        if which == 1:
            p1score[0] += (i - offset) + 4
            offset = i + 4
        else:
            p2score[0] += (i - offset) + 4
            offset = i + 4

    if p1score[0] == p2score[0]:
        draw[0] += p1score[0] + p2score[0] + 1


def winner_counts_for_pair(list decks_bytes, str p1, str p2, bint aligned=False, bint score_by_tricks=True) -> np.int64_t[:]:
    """
    decks_bytes: list of bytes objects (binary deck strings)
    p1, p2: pattern strings, length must be 3 or 4 (and p1/p2 must match lengths)

    aligned=True assumes decks are concatenated fixed-width patterns (3 or 4 bytes),
    which provides a small speedup by only checking starts at those boundaries.
    since patterns can start at any char boundary though, default is aligned=False.

    returns an array of [count_p1, count_p2, count_draw] as int64s
    """
    _init_simd()
    cdef bytes p1b = p1.encode("ascii")
    cdef bytes p2b = p2.encode("ascii")
    cdef Py_ssize_t w1 = PyBytes_GET_SIZE(p1b)
    cdef Py_ssize_t w2 = PyBytes_GET_SIZE(p2b)

    cdef const uint8_t* p1s = <const uint8_t*> PyBytes_AS_STRING(p1b)
    cdef const uint8_t* p2s = <const uint8_t*> PyBytes_AS_STRING(p2b)

    cdef uint32_t p1t
    cdef uint32_t p2t
    if w1 == 3:
        p1t = pack3(p1s, 0)
        p2t = pack3(p2s, 0)
    else:
        p1t = pack4(p1s, 0)
        p2t = pack4(p2s, 0)

    cdef long c0 = 0
    cdef long c1 = 0
    cdef long c2 = 0
    cdef Py_ssize_t k, m = len(decks_bytes)
    cdef bytes db
    cdef const uint8_t** ptrs = NULL
    cdef Py_ssize_t* sizes = NULL
    cdef const uint8_t* s
    cdef Py_ssize_t n
    cdef long p1score, p2score, draw

    score_by_cards = not score_by_tricks

    ptrs = <const uint8_t**>malloc(m * sizeof(const uint8_t*))
    sizes = <Py_ssize_t*>malloc(m * sizeof(Py_ssize_t))
    if ptrs == NULL or sizes == NULL:
        if ptrs != NULL:
            free(ptrs)
        if sizes != NULL:
            free(sizes)
        raise MemoryError()

    for k in range(m):
        db = <bytes>decks_bytes[k]
        ptrs[k] = <const uint8_t*>PyBytes_AS_STRING(db)
        sizes[k] = PyBytes_GET_SIZE(db)

    with nogil:
        if w1 == 3:
            for k in range(m):
                s = ptrs[k]
                n = sizes[k]
                score_one3(s, n, p1t, p2t, aligned, &p1score, &p2score, &draw)

                if score_by_cards:
                    c0 += p1score
                    c1 += p2score
                    c2 += draw
                else:
                    if p1score >= p2score and p1score >= draw:
                        c0 += 1
                    elif p2score >= p1score and p2score >= draw:
                        c1 += 1
                    else:
                        c2 += 1
        else:
            for k in range(m):
                s = ptrs[k]
                n = sizes[k]
                score_one4(s, n, p1t, p2t, aligned, &p1score, &p2score, &draw)

                if score_by_cards:
                    c0 += p1score
                    c1 += p2score
                    c2 += draw
                else:
                    if p1score >= p2score and p1score >= draw:
                        c0 += 1
                    elif p2score >= p1score and p2score >= draw:
                        c1 += 1
                    else:
                        c2 += 1

    free(ptrs)
    free(sizes)

    return np.array([c0, c1, c2], dtype=np.int64)
