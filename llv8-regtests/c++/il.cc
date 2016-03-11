#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <climits>
#include <iomanip>
using namespace std;

constexpr int ITER = 10;

int main() {
    int n, t;
    double p = 0.18;
    n = 1000; t = 1000;
    double expectation = 0;

    for (int ii = 0; ii < ITER; ii++) {
        vector<vector<double>> P; 
        P.resize(t + 1);
        for (int i = 0; i <= t; i++) {
            P[i].resize(n + 1);
        }
        for (int j = 1; j <= n; j++)
            P[0][j] = 0;
        P[0][0] = 1;

        for (int i = 1; i <= t; i++) {
            for (int k = 0; k <= n; k++) {
                P[i][k] = P[i - 1][k] * (k == n ? 1 : (1 - p)) + 
                    (k == 0 ? 0 :  P[i - 1][k - 1] * p);
            }
        }

        for (int k = 0; k <= n; k++) 
            expectation += k * P[t][k];
    }
    cout << setprecision(7) << expectation << endl;
    return 0;
}
