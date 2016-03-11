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

bool found = false;
constexpr int inf = INT_MAX / 2;
int answer = inf;

void driver(int bear_a, int bear_b, vector<int>& dimensions) {
    vector<int> power_2(1, 1);
    vector<int> power_3(1, 1);
    vector<int> power_5(1, 1);
    for (int i = 1; i < max(dimensions[0], dimensions[0 + 3]); i++)
        power_2.push_back(power_2[power_2.size() - 1] * 2);
    for (int i = 1; i < max(dimensions[1], dimensions[1 + 3]); i++)
        power_3.push_back(power_3[power_3.size() - 1] * 3);
    for (int i = 1; i < max(dimensions[2], dimensions[2 + 3]); i++)
        power_5.push_back(power_5[power_5.size() - 1] * 5);

    answer = inf;
    for (int i1 = dimensions[0] - 1; i1 >= 0; i1--) {
        for (int i2 = dimensions[1] - 1; i2 >= 0; i2--) {
            for (int i3 = dimensions[2] - 1; i3 >= 0; i3--) {
                for (int j1 = dimensions[3] - 1; j1 >= 0; j1--) {
                    for (int j2 = dimensions[4] - 1; j2 >= 0; j2--) {
                        for (int j3 = dimensions[5] - 1; j3 >= 0; j3--) {
                            if (i1 + i2 + i3 + j1 + j2 + j3 == 0) continue;
                            if (bear_a / power_2[i1] / power_3[i2] / power_5[i3] == bear_b / power_2[j1] / power_3[j2] / power_5[j3] ) {
                                found = true;
                                answer = min(answer, i1 + i2 + i3 + j1 + j2 + j3);
                            }
                        }
                    }
                }
            }
        }
    }
}

int main() {
    vector<int> aa = 
#include "/tmp/aa.cc"
     ;
    vector<int> bb = 
#include "/tmp/aa.cc"
     ;

    long long mega_ans = 0;
    for (int a : aa) {
        for (int b : bb) {
            found = false;
            answer = inf;

            if (a == b) {
                continue;
            }
            vector<int> powers; // pair<bear = a|b, prime = 2|3|5> -> degree of prime in bear
            vector<int> dimensions;

            for (auto bear : { a, b }) {
                auto bear_leftover = bear;
                for (auto power : { 2, 3, 5} ) {
                   powers.push_back(0);
                   while (bear_leftover % power == 0) {
                     powers[powers.size() - 1]++;
                     bear_leftover /= power;
                   }
                   dimensions.push_back(powers[powers.size() - 1] + 1);
                }
            }

            driver(a, b, dimensions);
            if (!found) answer = -1;
            mega_ans += answer;
        }
    }
    cout << mega_ans << endl;
    return 0;
}
