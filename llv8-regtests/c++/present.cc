#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <climits>
using namespace std;

constexpr long long int BILLION = 1000LL * 1000 * 1000; 
constexpr long long int K_100 = 100LL * 1000; 

using bitvector = vector<uint8_t>;

// Can the height of the shortest flower reach <height>
static bool can_grow(vector<int>& a, int w, int days, int height) {
    vector<int> ends(a.size() + w + 1, 0);
    int current = 0;
    int total = 0;
    for (int i = 0; i < a.size(); i++) {
        current -= ends[i];
        int delta = max(0, height - (a[i] + current));
        current += delta;
        ends[i + w] += delta;
        total += delta;
        if (total > days) return false;
    }
    return true;
}

int main() {
    long long n, days, w;
    cin >> n >> days >> w;
    vector<int> a(n);
    int min_height = BILLION + 2 * K_100;
    int max_height = BILLION + 2 * K_100; // The answeer cannot be greater than that.
    for (int i = 0; i < n; i++) {
        cin >> a[i];
        min_height = min(min_height, a[i]);
    }
    int l = min_height;  // [l, r)
    int r = max_height + 1;
    while (r - l > 1) {
        int m = (r + l) / 2;
        if (can_grow(a, w, days, m))
            l = m;
        else
            r = m;
    }
    cout << l << endl;
    return 0;
}
