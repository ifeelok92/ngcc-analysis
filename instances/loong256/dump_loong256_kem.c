#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "KEM_Loong.h"
#include "auxfunc.h"
#include "drng.h"
#include "params.h"
#include "poly.h"

DRNG_ctx drng_algorithm;

static void print_hex(const char *name, const unsigned char *buf, size_t len) {
    printf("%s ", name);
    for (size_t i = 0; i < len; i++) printf("%02x", buf[i]);
    printf("\n");
}

int main(int argc, char **argv) {
    unsigned char seed[64];
    unsigned char fill = 0x42;
    if (argc > 1) fill = (unsigned char)strtoul(argv[1], NULL, 0);
    memset(seed, fill, sizeof(seed));
    init_random_number(&drng_algorithm, seed, sizeof(seed));

    unsigned char pk[PUBLICKEY_BYTES], sk[SECRETKEY_BYTES];
    unsigned char ct[CIPHERTEXT_BYTES], ss[SHARED_KEY_BYTES];
    unsigned long long pk_len = 0, sk_len = 0, ct_len = 0, ss_len = 0;
    if (kem_keygen(pk, &pk_len, sk, &sk_len) != 0) return 1;
    if (kem_enc(pk, pk_len, ss, &ss_len, ct, &ct_len) != 0) return 2;

    size_t xlen = K1*K1*N + K1*K2*N + K2*K1*N + K2*K2*N*N;
    uint8_t *xbuf = calloc(3*xlen, 1);
    int16_t *A = calloc(xlen, sizeof(int16_t));
    if (!xbuf || !A) return 3;
    if (pseudoXOF(3*8*xlen, pk, SEED_BYTES*8, xbuf) != 0) return 4;
    sample_vector(xbuf, A, xlen);

    int16_t Bc[K2*N*N], B[K2*N*N];
    int16_t Uc[K2*N*N], U[K2*N*N];
    int16_t Vc[N*N], V[N*N];
    decode_vector(pk + SEED_BYTES + (K1*N*DB/8), Bc, K2*N*N, DB);
    decompress(Bc, K2*N*N, B, QBITS-DB);
    decode_vector(ct + K1*N*DU/8, Uc, K2*N*N, DU);
    decompress(Uc, K2*N*N, U, QBITS-DU);
    decode_vector(ct + (K1*N+K2*N*N)*DU/8, Vc, N*N, DV);
    decompress(Vc, N*N, V, QBITS-DV);

    int a4off = K1*K1*N + 2*K1*K2*N;
    printf("N %d K1 %d K2 %d Q %d ETA %d DU %d DV %d\n", N, K1, K2, Q, ETA, DU, DV);
    print_hex("PK", pk, PUBLICKEY_BYTES);
    print_hex("CT", ct, CIPHERTEXT_BYTES);
    print_hex("SS", ss, SHARED_KEY_BYTES);
    printf("A4");
    for (int i = 0; i < K2*N*K2*N; i++) printf(" %d", A[a4off+i]);
    printf("\nB");
    for (int i = 0; i < K2*N*N; i++) printf(" %d", B[i]);
    printf("\nU");
    for (int i = 0; i < K2*N*N; i++) printf(" %d", U[i]);
    printf("\nV");
    for (int i = 0; i < N*N; i++) printf(" %d", V[i]);
    printf("\n");

    free(xbuf);
    free(A);
    return 0;
}
