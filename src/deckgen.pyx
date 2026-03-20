# cython: language_level=3
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False

from libc.stdint cimport uint32_t
from libc.stddef cimport size_t
from libc.stdint cimport uintptr_t
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.unicode cimport PyUnicode_FromStringAndSize


IF UNAME_SYSNAME == "Linux":
    from libc.errno cimport errno
    cdef extern from "sys/random.h":
        ctypedef long ssize_t
        ssize_t getrandom(void *buf, size_t buflen, unsigned int flags) nogil

cdef extern from "time.h":
    long time(long* tloc) nogil

cdef extern from "unistd.h":
    int getpid() nogil


IF UNAME_SYSNAME == "Darwin":
    cdef extern from "stdlib.h":
        uint32_t arc4random_uniform(uint32_t upper_bound) nogil
ELSE:
    cdef extern from "stdlib.h":
        int rand_r(unsigned int *seedp) nogil


cdef inline unsigned int _seed32() nogil:
    cdef unsigned int seed
    seed = <unsigned int>(<uintptr_t>&seed)

    IF UNAME_SYSNAME == "Linux":
        cdef ssize_t got
        got = getrandom(&seed, sizeof(seed), 0)
        if got == <ssize_t>sizeof(seed):
            return seed

    seed ^= <unsigned int>time(NULL)
    seed ^= <unsigned int>getpid()
    return seed


IF UNAME_SYSNAME == "Darwin":
    cdef inline uint32_t _randbelow(uint32_t n) nogil:
        if n <= 1:
            return 0
        return arc4random_uniform(n)
ELSE:
    cdef inline uint32_t _rand_u31(unsigned int* seedp) nogil:
        return <uint32_t>rand_r(seedp)

    cdef inline uint32_t _randbelow(unsigned int* seedp, uint32_t n) nogil:
        # rejection sampling to avoid modulo bias: https://romailler.ch/2020/07/28/crypto-modulo_bias_guide/
        cdef uint32_t r
        cdef uint32_t threshold
        if n <= 1:
            return 0
        threshold = (<uint32_t>(-n)) % n
        while True:
            r = _rand_u31(seedp)
            if r >= threshold:
                return r % n


cpdef list generate_deck_strings(int num_decks, int deck_size):
    """
    generate random decks containing only zeros and ones, returned as list of 0/1 strings
    """
    cdef int half
    cdef int d, i, j
    cdef unsigned char* deck = NULL
    cdef char* out = NULL
    cdef list decks
    cdef unsigned char tmp
    IF not (UNAME_SYSNAME == "Darwin"):
        cdef unsigned int rng_state

    half = deck_size // 2
    decks = [None] * num_decks

    deck = <unsigned char*>PyMem_Malloc(deck_size)
    out = <char*>PyMem_Malloc(deck_size)
    if deck == NULL or out == NULL:
        if deck != NULL:
            PyMem_Free(deck)
        if out != NULL:
            PyMem_Free(out)
        raise MemoryError()

    # initialize once
    for i in range(half):
        deck[i] = 0
    for i in range(half, deck_size):
        deck[i] = 1
    IF not (UNAME_SYSNAME == "Darwin"):
        rng_state = _seed32()

    try:
        for d in range(num_decks):
            with nogil:
                # fisher-yates shuffle: https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
                for i in range(deck_size - 1, 0, -1):
                    IF UNAME_SYSNAME == "Darwin":
                        j = <int>_randbelow(<uint32_t>(i + 1))
                    ELSE:
                        j = <int>_randbelow(&rng_state, <uint32_t>(i + 1))
                    tmp = deck[i]
                    deck[i] = deck[j]
                    deck[j] = tmp

                # map numbers to strings
                for i in range(deck_size):
                    out[i] = <char>(48 + deck[i])

            decks[d] = PyUnicode_FromStringAndSize(out, deck_size)
    finally:
        PyMem_Free(deck)
        PyMem_Free(out)

    return decks
