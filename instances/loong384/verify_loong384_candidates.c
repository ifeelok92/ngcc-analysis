#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "KEM_Loong.c"

DRNG_ctx drng_algorithm;

static int hexval(int c) {
    if ('0' <= c && c <= '9') return c - '0';
    if ('a' <= c && c <= 'f') return c - 'a' + 10;
    if ('A' <= c && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int read_hex_bytes(const char *s, unsigned char *out, size_t len) {
    while (*s && isspace((unsigned char)*s)) s++;
    for (size_t i = 0; i < len; i++) {
        int hi = hexval((unsigned char)s[2*i]);
        int lo = hexval((unsigned char)s[2*i+1]);
        if (hi < 0 || lo < 0) return -1;
        out[i] = (unsigned char)((hi << 4) | lo);
    }
    return 0;
}

static void print_hex(const char *name, const unsigned char *buf, size_t len) {
    printf("%s ", name);
    for (size_t i = 0; i < len; i++) printf("%02x", buf[i]);
    printf("\n");
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;

    unsigned char pk[PUBLICKEY_BYTES], ct[CIPHERTEXT_BYTES], expected[SHARED_KEY_BYTES];
    int have_pk = 0, have_ct = 0, have_ss = 0;
    FILE *dump = fopen(argv[1], "r");
    if (!dump) return 3;
    char line[32768];
    while (fgets(line, sizeof(line), dump)) {
        if (!strncmp(line, "PK ", 3)) have_pk = !read_hex_bytes(line + 3, pk, PUBLICKEY_BYTES);
        if (!strncmp(line, "CT ", 3)) have_ct = !read_hex_bytes(line + 3, ct, CIPHERTEXT_BYTES);
        if (!strncmp(line, "SS ", 3)) have_ss = !read_hex_bytes(line + 3, expected, SHARED_KEY_BYTES);
    }
    fclose(dump);
    if (!have_pk || !have_ct) return 4;

    unsigned char h[HASH_BYTES];
    if (sm3hash(HASH_BYTES*8, pk, PUBLICKEY_BYTES*8, h) != 0) return 5;

    FILE *cand = fopen(argv[2], "r");
    if (!cand) return 6;
    unsigned long long checked = 0;
    while (fgets(line, sizeof(line), cand)) {
        unsigned char msg[MSG_BYTES], buf[MSG_BYTES+HASH_BYTES];
        unsigned char kg[SHARED_KEY_BYTES+3*SEED_BYTES], trial[CIPHERTEXT_BYTES];
        if (read_hex_bytes(line, msg, MSG_BYTES) != 0) continue;
        memcpy(buf, msg, MSG_BYTES);
        memcpy(buf + MSG_BYTES, h, HASH_BYTES);
        if (pseudoXOF((SHARED_KEY_BYTES+3*SEED_BYTES)*8, buf, (MSG_BYTES+HASH_BYTES)*8, kg) != 0) return 7;
        indcpa_encrypt(msg, pk, kg + SHARED_KEY_BYTES, trial);
        checked++;
        if (!memcmp(trial, ct, CIPHERTEXT_BYTES)) {
            printf("FOUND checked=%llu\n", checked);
            print_hex("MSG", msg, MSG_BYTES);
            print_hex("SS", kg, SHARED_KEY_BYTES);
            if (have_ss) printf("SS_MATCH %s\n", !memcmp(kg, expected, SHARED_KEY_BYTES) ? "true" : "false");
            fclose(cand);
            return 0;
        }
    }
    fclose(cand);
    printf("NOT_FOUND checked=%llu\n", checked);
    return 1;
}
