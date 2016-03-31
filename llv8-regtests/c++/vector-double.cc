#include <iostream>
#include <vector>

constexpr int ITER = 1000;

int main() {
    int n, t;
    double p = 0.18;
    n = 1000;
    t = 5;

    std::vector<double> P(n);
    for (int ii = 0; ii < ITER; ii++) {
        P[0] = p;
        for (int j = 1; j < n; j++)
            P[j] = P[j - 1] * P[j - 1];
    }
    std::cout << P[t] << std::endl;
    return 0;
}
