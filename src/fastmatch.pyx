# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_GET_SIZE
from libc.stdint cimport uint32_t, uint8_t
import numpy as np
cimport numpy as cnp


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

    p1score[0] = 0
    p2score[0] = 0
    draw[0] = 0
    if n < 3:
        draw[0] = 1
        return

    while offset <= n - 3:
        i = offset
        found = False
        while i <= n - 3:
            tri = pack3(s, i)
            if tri == p1t:
                p1score[0] += (i - offset) + 3
                offset = i + 3
                found = True
                break
            elif tri == p2t:
                p2score[0] += (i - offset) + 3
                offset = i + 3
                found = True
                break
            i += step
        if not found:
            draw[0] += 1
            break

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

    p1score[0] = 0
    p2score[0] = 0
    draw[0] = 0
    if n < 4:
        draw[0] = 1
        return

    while offset <= n - 4:
        i = offset
        found = False
        while i <= n - 4:
            quad = pack4(s, i)
            if quad == p1t:
                p1score[0] += (i - offset) + 4
                offset = i + 4
                found = True
                break
            elif quad == p2t:
                p2score[0] += (i - offset) + 4
                offset = i + 4
                found = True
                break
            i += step
        if not found:
            draw[0] += 1
            break

    if p1score[0] == p2score[0]:
        draw[0] += p1score[0] + p2score[0] + 1


def winner_counts_for_pair(list decks_bytes, str p1, str p2, bint aligned=False, bint score_by_cards=False) -> np.int64_t[:]:
    """
    decks_bytes: list of bytes objects (binary deck strings)
    p1, p2: pattern strings, length must be 3 or 4 (and p1/p2 must match lengths)

    aligned=True assumes decks are concatenated fixed-width patterns (3 or 4 bytes),
    which provides a small speedup by only checking starts at those boundaries.
    since patterns can start at any char boundary though, default is aligned=False.

    returns an array of [count_p1, count_p2, count_draw] as int64s
    """
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
    cdef const uint8_t* s
    cdef Py_ssize_t n
    cdef long p1score, p2score, draw

    if w1 == 3:
        for k in range(m):
            db = <bytes>decks_bytes[k]
            s = <const uint8_t*> PyBytes_AS_STRING(db)
            n = PyBytes_GET_SIZE(db)
            with nogil:
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
            db = <bytes>decks_bytes[k]
            s = <const uint8_t*> PyBytes_AS_STRING(db)
            n = PyBytes_GET_SIZE(db)
            with nogil:
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

    return np.array([c0, c1, c2], dtype=np.int64)
