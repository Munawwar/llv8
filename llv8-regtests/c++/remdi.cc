#include <iostream>
#include <cstdint>

// Flags: --llvm-filter=-_rand

constexpr int MOD = 1000000007;
constexpr int STEPS = 50000* 1000 * 10;

int main() {
    int total_total = 0;
    for (int j = 0; j < STEPS; j++) 
        total_total = (total_total + j) % MOD;
    std::cout << total_total;
    return 0;
}
