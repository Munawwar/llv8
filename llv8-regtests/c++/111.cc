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

constexpr long long int BILLION = 1000LL * 1000 * 1000; 

using bitvector = vector<uint8_t>;

int main() {
    long long n, x, y;
//    cin >> n >> x >> y;
    n = 999990;
    x = 9990000000;
    y = 9995000;
    vector<long long>a(n, 1);
    long long sum = n, squares_sum = n;
    // We want sum <= y, squares_sum >= x
    int pos = 0; // always increase a[0]
    while (squares_sum < x) {
        if (sum > y) break;
        auto was = a[pos];
        squares_sum -= was * was;
        sum -= was;
        a[pos] = was + 1;
        squares_sum += a[pos] * a[pos];
        sum += a[pos];
    }
    cout << squares_sum << endl;
    return 0;
}
