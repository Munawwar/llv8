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

enum color : int { RED, BLUE };
enum player : int { PETYA, VASYA };

void play(int last, int turn, int num_red, int num_blue, int& pe_total, int& vas_total) {
    if (num_red + num_blue == 0) return;
    if (num_red == 0) {
        pe_total += (BLUE == last);
        vas_total += (BLUE != last);
        return play(BLUE, 1 - turn, num_red, num_blue - 1, pe_total, vas_total);
    }
    if (num_blue == 0) {
        pe_total += (RED == last);
        vas_total += (RED != last);
        return play(RED, 1 - turn, num_red - 1, num_blue, pe_total, vas_total);
    }
    if (turn == VASYA) {
        // It is alaways beneficial for Vasya to change the color.
        vas_total += 1;
        return play(1 - last, 1 - turn, num_red - (last == BLUE), num_blue - (last == RED), pe_total, vas_total);
    } else {
        pe_total += 1;
        return play(last, 1 - turn, num_red - (last == RED), num_blue - (last == BLUE), pe_total, vas_total);
    }
}

int main() {
    long long n, m;
    //cin >> n >> m;
    n = 2952201;
    m = 2952200;
    int pe = 0, va = 0, pe2 = 0, va2 = 0;
    play(RED, VASYA, n - 1, m, pe, va);
    play(BLUE, VASYA, n, m - 1, pe2, va2);
    pair<int, int> ans;
    if (pe2 == pe) {
        ans = make_pair(pe, min(va, va2));
    } else {
        ans.first = max(pe, pe2);
        if (pe > pe2)
            ans.second = va;
        else
            ans.second = va2;
    }
    cout << ans.first << " " << ans.second << endl;
    return 0;
}
