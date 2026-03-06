#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>
#include "manager.h"
#include "spec_decode.h"

void print_usage() {
    printf("Usage: ./main [options]\n");
    printf("  --mode none|ngram|tardis   Decoding mode (default: none)\n");
    printf("  --prompt \"text\"            Prompt text\n");
    printf("  --max-tokens N             Tokens to generate (default: 1000)\n");
    printf("  --branch-factor B          Draft tree branching factor (default: 2)\n");
    printf("  --max-depth d              Draft tree max depth (default: 4)\n");
    printf("  --ngram-size k             Max n-gram size (default: 4)\n");
    printf("  --tardis-dim D             TARDIS embedding dim (default: 64)\n");
    printf("  --alpha A                  TARDIS alpha (default: 0.1)\n");
    printf("  --beta B                   TARDIS beta (default: 0.1)\n");
    printf("  --omega W                  TARDIS omega (default: pi/4)\n");
    printf("  --min-sim S                TARDIS min similarity (default: 0.3)\n");
    printf("  --min-dir D                TARDIS min directionality (default: 0.1)\n");
    printf("  --phi P                    TARDIS scoring blend: cos(phi)*sim + sin(phi)*dir\n");
    printf("  --sim-threshold T          TARDIS sim threshold filter\n");
}

int main(int argc, char** argv) {
    SpecDecodeConfig cfg;
    std::string prompt = "Hello, my name is llama and I am a friendly assistant running on Aryan's GPU. My favorite flavor of ice cream is";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "none") == 0) cfg.mode = SpecDecodeConfig::NONE;
            else if (strcmp(argv[i], "ngram") == 0) cfg.mode = SpecDecodeConfig::NGRAM;
            else if (strcmp(argv[i], "tardis") == 0) cfg.mode = SpecDecodeConfig::TARDIS;
            else { printf("Unknown mode: %s\n", argv[i]); return 1; }
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            cfg.num_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--branch-factor") == 0 && i + 1 < argc) {
            cfg.branch_factor = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max-depth") == 0 && i + 1 < argc) {
            cfg.max_depth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ngram-size") == 0 && i + 1 < argc) {
            cfg.max_ngram_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tardis-dim") == 0 && i + 1 < argc) {
            cfg.tardis_dim = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) {
            cfg.alpha = atof(argv[++i]);
        } else if (strcmp(argv[i], "--beta") == 0 && i + 1 < argc) {
            cfg.beta = atof(argv[++i]);
        } else if (strcmp(argv[i], "--omega") == 0 && i + 1 < argc) {
            cfg.omega = atof(argv[++i]);
        } else if (strcmp(argv[i], "--min-sim") == 0 && i + 1 < argc) {
            cfg.min_sim = atof(argv[++i]);
        } else if (strcmp(argv[i], "--min-dir") == 0 && i + 1 < argc) {
            cfg.min_dir = atof(argv[++i]);
        } else if (strcmp(argv[i], "--phi") == 0 && i + 1 < argc) {
            cfg.phi = atof(argv[++i]);
        } else if (strcmp(argv[i], "--sim-threshold") == 0 && i + 1 < argc) {
            cfg.sim_threshold = atof(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage();
            return 0;
        } else {
            printf("Unknown argument: %s\n", argv[i]);
            print_usage();
            return 1;
        }
    }

    const char* mode_names[] = {"none", "ngram", "tardis"};
    printf("Mode: %s, Max tokens: %d\n", mode_names[(int)cfg.mode], cfg.num_tokens);
    if (cfg.mode == SpecDecodeConfig::NGRAM) {
        printf("N-gram: size=%d, branch=%d, depth=%d\n", cfg.max_ngram_size, cfg.branch_factor, cfg.max_depth);
    } else if (cfg.mode == SpecDecodeConfig::TARDIS) {
        printf("TARDIS: dim=%d, alpha=%.2f, beta=%.2f, omega=%.2f, branch=%d, depth=%d",
               cfg.tardis_dim, cfg.alpha, cfg.beta, cfg.omega, cfg.branch_factor, cfg.max_depth);
        if (!std::isnan(cfg.phi)) printf(", phi=%.3f", cfg.phi);
        if (!std::isnan(cfg.sim_threshold)) printf(", sim_thresh=%.3f", cfg.sim_threshold);
        printf("\n");
    }

    Manager manager;
    std::cout << "Calling handle_request on: " << prompt << std::endl;
    std::optional<std::string> response = manager.handle_request(prompt, cfg);
    if (response.has_value()) {
        std::cout << "Response: " << response.value() << std::endl;
    } else {
        std::cout << "No response" << std::endl;
    }
}
