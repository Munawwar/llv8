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


int power(int x, int n, int mod) {
    int cur = 1;
    while (n != 0) {
        if (n % 2 != 0)
            cur = (cur * x) % mod;
        x = x * x;
        n = n / 2;
    }
    return cur;
}

int factorial(int n) {
    int cur = 1;
    while (n > 1) {
        cur *= n;
        n--;
    }
    return cur;
}

int main() {
    int n, M, k, N;
    N = 100000 * 100;
    vector<int> c = { -1, 345, 451, 12516, 1478, -675, -57854, -567, -65582, 22, -2134  };
    M = c.size();
    int sum = 0;
    int x = 22;
    int MOD = 10000007;
    for (n = 0; n < N; n++) {
        int k = n % M;
        sum += (c[k] * power(x, n, MOD) + n / factorial(k)) % MOD;
    }
    cout << sum << endl;
    return 0;
}
