#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cstdint>
using namespace std;

constexpr long long int BILLION = 1000LL * 1000 * 1000; 

using bitvector = vector<uint8_t>;


int main() {
    long long n;

    vector<int> input_vector = 
    
#include "/tmp/5.cc"
    ;
     
    n = input_vector.size();

    vector<int> a(n+1);
    a[0] = -1; // unused
    map<int, int> where;

    for (int i = 1; i <= n; i++) {
        int next = input_vector[i - 1];
        a[i] = next;
        where[next] = i;
    }
    // While not empty: find the max, 
    // if it is not the last element, then 
    //    everything before it has updated. => 
    //    Discard it and the max and add the number of discarded elements to the answer.
    // else
    //    simply discard the max.
    int ans = 0;
    int first_index = 1, last_index = n;
    while (last_index > first_index) {
        cout << first_index << " " << last_index << endl;
        int max_value = where.rbegin()->first;     // The max. 
        int max_position = where.rbegin()->second; // So max_position is the slot where the last (among not discarded) old theread is at.
        if (max_position != last_index) {
            for (int index = first_index; index <= max_position; index++) {
                ans++;
                where.erase(a[index]);
            }
            first_index = max_position + 1;
        } else {
            last_index--;
        }
        where.erase(max_value);
    }
    cout << ans << endl;
    return 0;
}
