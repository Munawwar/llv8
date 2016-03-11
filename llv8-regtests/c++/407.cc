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

// Flags: --llvm-filter=-_rand

constexpr long long int BILLION = 1000LL * 1000 * 1000; 
constexpr int MOD = 1000000007;
constexpr int STEPS = 5;

int saved_ans[10000];

int go_back_and_return(int v, vector<int>& p) {
   if (saved_ans[v] != -1) return saved_ans[v];
   int cur = p[v] - 1;
   int ans = 1;
   while (cur < v) {
       ans = (ans + go_back_and_return(cur, p)) % MOD;
       cur++;
       ans = (ans + 1) % MOD;
   }
   saved_ans[v] = ans;
   return ans;
}

vector<int> generate() {
    vector<int> res;
    int n = 9900;
    res.resize(n);
    for (int i = 0; i < n; i++) {
        auto room = i + 1;
        int p_room = rand() % room + 1;
        res[i] = p_room;
    }
    return res;
}

int main() {
    int n;
    vector<int> p; 
    srand(123);

    int total_total = 0;
    for (int k = 0; k < STEPS; k++) {
        p = generate();

        n = p.size();
        memset(saved_ans, 0xff, sizeof saved_ans); // -1
        int total = 0;
        // We entered i-th room for the first time, thus making the number of Xes on it's ceiling odd,
        // and numbers of Xes of all the rooms left from the i-th even.
        for (auto i = 0; i < n; i++) {
            total = (total + go_back_and_return(i, p)) % MOD;
            total = (total + 1) % MOD;
        }
        total_total = (total_total + total) % MOD;
    }
    cout << total_total;
    return 0;
}
