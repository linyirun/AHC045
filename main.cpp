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

// VERSION: K-means for initial clustering, with naive mst


// STRUCTS ----------------------------------------------
struct City;
struct Problem;

struct City {
    int lx{}, rx{}, ly{}, ry{};
    int idx{};  // index of the city w.r.t the problem

    ld cx{}, cy{};  // center of the coords
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


// TODO: add a class that just runs Kruskal's given cities


class Solver {
private:

    ld dist(ld x1, ld y1, ld x2, ld y2) {
        // computes the squared euclidean dist
        ld dx = x1 - x2;
        ld dy = y1 - y2;
        return dx * dx + dy * dy;
    }

    ld dist_center(int i, int j) {
        // Computes the squared Euclidean distance between city i and city j, using their city centers
        ld dx = problem.cities[i].cx - problem.cities[j].cx;
        ld dy = problem.cities[i].cy - problem.cities[j].cy;
        return dx * dx + dy * dy;
    }

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
        // vector<int> indices(problem.N);
        //
        // for (int i = 0; i < problem.N; i++) {
        //     indices[i] = i;
        // }
        //
        // // Randomly shuffle the array, then partition into those group sizes
        // shuffle(indices.begin(), indices.end(), rng);
        // vector<vector<int>> groups(problem.M);
        // //
        // int cnt = 0;
        // for (int i = 0; i < problem.M; i++) {
        //     groups[i].resize(problem.group_sizes[i]);
        //     for (int j = 0; j < problem.group_sizes[i]; j++) {
        //         groups[i][j] = indices[cnt++];
        //     }
        // }

        vector<vector<int>> groups = KMeans(100);

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

    vector<vector<int>> KMeans(int num_iters) {
        /* K-means approach for initial clustering, based off of estimated city centers
         *
         * Assumes that each city is in the center of its rectangle
         *
         * 1) Randomly select clusters, compute its centers
         * 2) Repeat num_iters times:
         *  2a) For each idx k cluster center, compute the G_k closest (unselected) points
         *  2b) Compute the new cluster center based off of this
         *
         *
         * Returns:
         * size (M, G_k) 2d vec of city index assignments
         */


        // 1) Randomly select clusters: --------------------------------------------------
        vector<int> indices(problem.N);
        for (int i = 0; i < problem.N; i++) {
            indices[i] = i;
        }

        // Randomly shuffle the array, then partition into those group sizes
        shuffle(indices.begin(), indices.end(), rng);
        vector<vector<int>> prev_groups(problem.M);
        vector<pair<ld, ld>> prev_group_centers(problem.M);
        int cnt = 0;
        for (int i = 0; i < problem.M; i++) {
            ld center_x = 0, center_y = 0;
            prev_groups[i].resize(problem.group_sizes[i]);
            for (int j = 0; j < problem.group_sizes[i]; j++) {
                int city_idx = indices[cnt++];
                prev_groups[i][j] = city_idx;
                // Compute group centers
                center_x += problem.cities[city_idx].cx;
                center_y += problem.cities[city_idx].cy;
            }
            prev_group_centers[i] = {center_x / problem.group_sizes[i], center_y / problem.group_sizes[i]};
        }

        // 2) Main K-means loop ----------------------------------------------------------
        for (int iter = 0; iter < num_iters; iter++) {
            vector<bool> taken_cities(problem.N); // cities that have already been selected
            vector<vector<int>> curr_groups = prev_groups; // The current group assignments
            vector<pair<ld, ld>> curr_group_centers(problem.M);

            for (int group_idx = 0; group_idx < problem.M; group_idx++) {
                vector<pair<ld, int>> distances;
                // Find the G_k closest points
                for (int i = 0; i < problem.N; i++) {
                    if (taken_cities[i]) continue;
                    // Push back {dist, city idx}
                    distances.push_back({
                        dist(problem.cities[i].cx, problem.cities[i].cy, prev_group_centers[group_idx].first, prev_group_centers[group_idx].second), i
                    });
                }
                sort(distances.begin(), distances.end());

                // assign the G_k closest pairs, also recompute the cluster centers
                ld center_x = 0, center_y = 0;
                for (int i = 0; i < problem.group_sizes[group_idx]; i++) {
                    int city_idx = distances[i].second;
                    taken_cities[city_idx] = true;
                    curr_groups[group_idx][i] = city_idx;

                    // recompute cluster centers
                    center_x += problem.cities[city_idx].cx;
                    center_y += problem.cities[city_idx].cy;
                }

                // Update cluster centers
                curr_group_centers[group_idx] = {center_x / problem.group_sizes[group_idx], center_y / problem.group_sizes[group_idx]};
            }

            prev_groups = curr_groups; // TODO: do I even need to store the prev group assignments?
            // TODO: isn't it enough to just store the final one?
            prev_group_centers = curr_group_centers;
        }

        return prev_groups;
    }

    void MCMF() {
        // TODO: consider this more - is it even worth it?
        /* Min Cost Max Flow + K-means
         * This computes groups _without_ using any MST queries.
         * Used as a first step to compute the groups
         *
         *
         * Approach:
         * Randomly initialize cluster centers
         * Repeat:
         *   Use MCMF to find the clusters based off dist to mean
         *
         *
         *
         */
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
        problem.cities[i].cx = (ld) (problem.cities[i].lx + problem.cities[i].rx) / 2;
        problem.cities[i].cy = (ld) (problem.cities[i].ly + problem.cities[i].ry) / 2;
#else

#endif

        problem.cities[i].idx = i;


    }

    Solver solver;
    solver.solve();
}
