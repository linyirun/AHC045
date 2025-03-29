#include <bits/stdc++.h>
#define ll long long
#define ld long double
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

#ifdef DEBUG
#define dbg(x) cerr << "[ " << #x << " = " << (x) << " ]" << '\n';
#else
#define dbg(x)
#endif

// #define int ll
//const int MOD = 998244353;
// const int MOD = 1000000007;
// const int INF = 1e15;


// STRUCTS ----------------------------------------------
struct City;
struct Problem;

struct City {
    int lx{}, rx{}, ly{}, ry{};
    int idx{};  // index of the city w.r.t the problem

    int cx{}, cy{};  // center of the coords
};

struct Problem {
    int N{}, M{}, Q{}, L{}, W{};
    vector<int> group_sizes;
    vector<City> cities;
};

struct State {

};



// GLOBAL VARS ------------------------------------------
Problem problem;  // problem data

random_device rd;
// mt19937 rng(rd());

// USING FIXED SEED:
mt19937 rng(0);


vector<pii> query(vector<int> &cities) {
    // queries cities and returns the results in a pair

    int l = cities.size();
    if (l == 1) return {}; // Possible to just have 1 city
    // assert(l > 1);

    cout << "? " << l << ' ';
    for (int i : cities) cout << i << ' ';
    cout << '\n';

    cout.flush();

    // Read in the result
    vector<pii> res(l - 1);
    for (int i = 0; i < l - 1; i++) {
        cin >> res[i].first >> res[i].second;
    }

    return res;
}

void answer(vector<vector<pii>> &edges) {

}



class Solver {
public:
    void solve() {
        naive_mst();
    }

    void setup() {
        // setup
    }

    void naive_mst() {
        // Naive sol:
        // Randomly select the groups, query the MSTs on them
        vector<int> indices(problem.N);

        for (int i = 0; i < problem.N; i++) {
            indices[i] = i;
        }

        // Randomly shuffle the array, then partition into those group sizes
        shuffle(indices.begin(), indices.end(), rng);
        vector<vector<int>> groups(problem.M);
        //
        int cnt = 0;
        for (int i = 0; i < problem.M; i++) {
            groups[i].resize(problem.group_sizes[i]);
            for (int j = 0; j < problem.group_sizes[i]; j++) {
                groups[i][j] = indices[cnt++];
            }
        }

        vector<vector<pii>> edges(problem.M);
        // Query, then save the edges of the MST
        // Query the next L


        for (int i = 0; i < problem.M; i++) {
            int prev_group_idx = -1; // One representative node in the prev group
            for (int j = 0; j < problem.group_sizes[i]; j += problem.L) {
                vector<int> to_query;
                for (int k = j; k < j + problem.L && k < problem.group_sizes[i]; k++) {
                    to_query.push_back(groups[i][k]);
                }
                // Query these and add it to edges
                vector<pii> ans = query(to_query);
                edges[i].insert(edges[i].end(), ans.begin(), ans.end());

                if (prev_group_idx != -1) {
                    // If there's a prev group, link this component to the prev
                    edges[i].push_back({prev_group_idx, groups[i][j]});
                }
                prev_group_idx = groups[i][j];
            }
        }

        // Report the edges

        cout << "!\n";
        for (int i = 0; i < problem.M; i++) {
            // print out groups
            for (int j = 0; j < problem.group_sizes[i]; j++) {
                cout << groups[i][j] << ' ';
            }
            cout << '\n';
            // print out edges
            for (pii &p : edges[i]) {
                cout << p.first << ' ' << p.second << '\n';
            }
        }
        cout.flush();

    }

};





int32_t main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    #ifdef DEBUG
    cerr << "Running in debug mode...\n";
    #endif

    // Read in data
    cin >> problem.N >> problem.M >> problem.Q >> problem.L >> problem.W;

    problem.group_sizes.resize(problem.M);
    problem.cities.resize(problem.N);

    for (int i = 0; i < problem.M; i++) {
        cin >> problem.group_sizes[i];
    }

    for (int i = 0; i < problem.N; i++) {
#ifndef DEBUG
        cin >> problem.cities[i].lx >> problem.cities[i].rx >> problem.cities[i].ly >> problem.cities[i].ry;
        // calculate the centers
        problem.cities[i].cx = (problem.cities[i].lx + problem.cities[i].rx) / 2;
        problem.cities[i].cy = (problem.cities[i].ly + problem.cities[i].ry) / 2;
#else

#endif

        problem.cities[i].idx = i;


    }

    Solver solver;
    solver.solve();
}
