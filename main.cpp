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

// VERSION: K-means for initial clustering with 2-opt, 0.9 threshold, with naive mst
// new: added mini-clustering within the groups too


// STRUCTS ----------------------------------------------
struct City;
struct Problem;

struct City {
    int lx{}, rx{}, ly{}, ry{};
    int idx{};  // index of the city w.r.t the problem
    int group_idx{}; // group that this city belongs to
    int group_idx_j{}; // the index within that group that contains this city

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


void answer(vector<vector<pii>> &edges) {

}


// TODO: add a class that just runs Kruskal's given cities


class Solver {
private:
    int queries_used = 0; // Number of queries used
    vector<pii> query(vector<int> &cities) {
        int l = cities.size();
        if (l == 1) return {}; // Possible to just have 1 city
        // TODO: see if we have to return this in lex order
        if (l == 2) return {{cities[0], cities[1]}};
        // assert(l > 1);

        // Queries cities and returns the results in a vector<pii>
        if (queries_used >= problem.Q) {
            cerr << "Max number of queries reached!!!\n";
            return {};
        }
        queries_used++;


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

    vector<int> generate_indices(int n) {
        // Gets indices from 0 to n - 1
        vector<int> a(n);
        for (int i = 0; i < n; i++) a[i] = i;
        return a;
    }

    pii generate_border_point(int BORDER_SIZE=10000) {
        // Generates a point on the border
        uniform_int_distribution<int> edge_gen(0, 3);
        uniform_real_distribution<ld> pt_gen(0, BORDER_SIZE);

        int edge = edge_gen(rng);
        ld loc = pt_gen(rng);

        pii pt = {0, 0};
        // 0: bottom, 1: left, 2: top, 3: right
        switch (edge) {
            case 0:
                pt = {loc, 0};
            case 1:
                pt = {0, loc};
            case 2:
                pt = {loc, BORDER_SIZE};
            case 3:
                pt = {BORDER_SIZE, loc};
        }
        return pt;

    }

public:

    void solve() {
        mst();

        cerr << "We have " << problem.Q - queries_used << " queries remaining to use, with L being " << problem.L << "\n";
    }

    void setup() {
        // setup
    }

    void mst() {
        // Generate MST

        vector<vector<int>> groups = KMeans(50, 10);

        vector<vector<pii>> edges(problem.M);
        // Query, then save the edges of the MST
        // Query the next L

        // WITH MINI-CLUSTERING --------------------------------------------------------
        vector<vector<vector<int>>> clustered_groups = miniClusteringKMeans(groups, 10);

        for (int i = 0; i < problem.M; i++) {
            int prev_group_idx = -1; // One representative node in the prev subset
            int num_clusters = clustered_groups[i].size();

            for (int curr_cluster_idx = 0; curr_cluster_idx < num_clusters; curr_cluster_idx++) {
                vector<int> to_query = clustered_groups[i][curr_cluster_idx]; // the set of cities in this mini-cluster
                // Query these and add it to edges
                vector<pii> ans = query(to_query);
                edges[i].insert(edges[i].end(), ans.begin(), ans.end());
                if (prev_group_idx != -1) {
                    // If there's a prev group, link this component to the prev
                    // edges[i].push_back({prev_group_idx, groups[i][j]});

                    // TODO: implement clustering MST instead of this

                    pii best_edge = {prev_group_idx, clustered_groups[i][curr_cluster_idx][0]};
                    ld best_dist = LDBL_MAX;

                    // Find the closest node in this cluster and any of the previous cluster, and join them
                    for (int prev_cluster_idx = 0; prev_cluster_idx < curr_cluster_idx; prev_cluster_idx++) {
                        for (int other_group_idx : clustered_groups[i][prev_cluster_idx]) {
                            for (int this_group_idx : clustered_groups[i][curr_cluster_idx]) {
                                ld new_dist = dist_center(this_group_idx, other_group_idx);
                                if (new_dist < best_dist) {
                                    best_dist = new_dist;
                                    best_edge = {this_group_idx, other_group_idx};
                                }
                            }
                        }
                    }

                    edges[i].push_back(best_edge);
                }
                prev_group_idx = clustered_groups[i][curr_cluster_idx][0];

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

    vector<vector<int>> KMeans(int num_iters, int num_2opt_iters=10) {
        /* K-means approach for initial clustering, based off of estimated city centers
         *
         * Assumes that each city is in the center of its rectangle
         *
         * 1) Randomly select clusters, compute its centers
         * 2) Repeat num_iters times:
         *  2a) For each idx k cluster center, compute the G_k closest (unselected) points
         *  2b) Compute the new cluster center based off of this
         * 3) Try 2-opt to improve clustering
         *
         * Returns:
         * size (M, G_k) 2d vec of city index assignments
         */


        // 1) Randomly select clusters: --------------------------------------------------
        vector<int> indices = generate_indices(problem.N);

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

            // Initialize to a random point on the border
            prev_group_centers[i] = generate_border_point();

        }



        // 2) Main K-means loop ----------------------------------------------------------
        for (int iter = 0; iter < num_iters; iter++) {
            vector<bool> taken_cities(problem.N); // cities that have already been selected
            vector<vector<int>> curr_groups = prev_groups; // The current group assignments
            vector<pair<ld, ld>> curr_group_centers(problem.M);

            // Loop over the group indexes in random order, to not give preference to earliest groups
            vector<int> group_indices = generate_indices(problem.M);
            shuffle(group_indices.begin(), group_indices.end(), rng);

            // Sort the group indices by the distance to its nearest border
            // vector<ld> dist_to_border(problem.M);
            // for (int i = 0; i < problem.M; i++) {
            //     pii center = prev_group_centers[i];
            //     dist_to_border[i] = min({center.first, center.second, 10000 - center.first, 10000 - center.second});
            // }
            //
            // sort(group_indices.begin(), group_indices.end(), [&dist_to_border](const int i1, const int i2) {
            //     return dist_to_border[i1] < dist_to_border[i2];
            // });


            for (int group_idx : group_indices) {
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

            bool done_flag = false;
            if (curr_groups == prev_groups) {
                done_flag = true;
                // cerr << "Done after " << iter << " iterations\n";
            }

            prev_groups = curr_groups; // TODO: do I even need to store the prev group assignments?
            // TODO: isn't it enough to just store the final one?
            prev_group_centers = curr_group_centers;

            if (done_flag) break;
        }




        // 4) 2-opt for groups
        // Calculate the total dist between a group and its center
        // vector<ld> total_dist_to_centers(problem.M);
        // for (int group_idx = 0; group_idx < problem.M; group_idx++) {
        //     for (int i = 0; i < problem.group_sizes[group_idx]; i++) {
        //         int city_idx = prev_groups[group_idx][i];
        //         // total_dist_to_centers[group_idx] += dist(problem.cities[city_idx].cx, problem.cities[city_idx].cy, prev_group_centers[group_idx].first, prev_group_centers[group_idx].second);
        //
        //         // Update the group index of each city too
        //         problem.cities[city_idx].group_idx = group_idx;
        //         // problem.cities[city_idx].group_idx_j = i;
        //     }
        // }
        //
        // for (int twoopt_iter = 0; twoopt_iter < num_2opt_iters; twoopt_iter++) {
        //     // TODO: loop through groups instead, not sure if it's worth it tho
        //     bool swapped = false;
        //     for (int city_idx1 = 0; city_idx1 < problem.N; city_idx1++) {
        //         for (int city_idx2 = city_idx1 + 1; city_idx2 < problem.N; city_idx2++) {
        //             int group1 = problem.cities[city_idx1].group_idx, group2 = problem.cities[city_idx2].group_idx;
        //             if (group1 == group2) continue;
        //             // Get respective distances
        //             ld dist1_to_group1 = dist(problem.cities[city_idx1].cx, problem.cities[city_idx1].cy, prev_group_centers[group1].first, prev_group_centers[group1].second);
        //             ld dist2_to_group2 = dist(problem.cities[city_idx2].cx, problem.cities[city_idx2].cy, prev_group_centers[group2].first, prev_group_centers[group2].second);
        //             ld dist1_to_group2 = dist(problem.cities[city_idx1].cx, problem.cities[city_idx1].cy, prev_group_centers[group2].first, prev_group_centers[group2].second);
        //             ld dist2_to_group1 = dist(problem.cities[city_idx2].cx, problem.cities[city_idx2].cy, prev_group_centers[group1].first, prev_group_centers[group1].second);
        //
        //             // If this swap strictly improves it, then swap these two
        //             if (dist1_to_group2 < 0.9 * dist1_to_group1 && dist2_to_group1 < 0.9 * dist2_to_group2) {
        //                 // Perform the swap
        //                 swapped = true;
        //                 // int group1_idx_j = problem.cities[city_idx1].group_idx_j, group2_idx_j = problem.cities[city_idx2].group_idx_j;
        //
        //                 // swap(problem.cities[city_idx1].group_idx_j, problem.cities[city_idx2].group_idx_j);
        //                 swap(problem.cities[city_idx1].group_idx, problem.cities[city_idx2].group_idx);
        //                 // cerr << "swapped city indices " << city_idx1 << " " << city_idx2 << '\n';
        //             }
        //         }
        //     }
        //
        //     if (!swapped) {
        //         cerr << "Stopped after " << twoopt_iter << " 2-opt iterations" << '\n';
        //         break;
        //     }
        // }
        //
        // vector<vector<int>> opt_groups(problem.M);
        // // Reconstruct the new groups after 2-opt
        // for (int city_idx = 0; city_idx < problem.N; city_idx++) {
        //     City* city = &problem.cities[city_idx];
        //     opt_groups[city->group_idx].push_back(city_idx);
        // }
        //
        // return opt_groups;
        // END OF 2-OPT ---------------------------------------------------

        return prev_groups;

    }

    vector<vector<vector<int>>> miniClusteringKMeans(vector<vector<int>> &groups, int kmeans_iters) {
        /* Given groups of size (M, G_k)
         *
         * For each group, perform K-means of clusters of size <= L.
         * This will have ceil(G_k / L) clusters:
         *   floor(G_k / L) mini-clusters of size L, 1 cluster of the rest of size G_k % L
         *
         * Runs the kmeans alg for kmeans_iters iterations
         *
         * Return a vector of:
         * [groups][num_clusters][group assignments]
         */

        // [groups][num_clusters][group assignments]
        vector<vector<vector<int>>> clustered_groups;
        clustered_groups.resize(problem.M);

        for (int group_idx = 0; group_idx < problem.M; group_idx++) {
            // all the city indices in this group
            // vector<int> indices = groups[group_idx];
            // shuffle(indices.begin(), indices.end(), rng);

            // Get the number of clusters and each clusters' size
            int num_clusters = (problem.group_sizes[group_idx] + problem.L - 1) / problem.L;
            vector<int> cluster_sizes(num_clusters);
            int total = problem.group_sizes[group_idx];
            for (int i = 0; i < num_clusters; i++) {
                cluster_sizes[i] = min(problem.L, total);
                total -= cluster_sizes[i];
            }

            clustered_groups[group_idx].resize(num_clusters);

            // prev_clusters has size [num_clusters][size of cluster]
            vector<vector<int>> prev_clusters(num_clusters);
            vector<pair<ld, ld>> prev_cluster_centers(num_clusters);

            // Randomly assign cluster centers
            int cnt = 0;
            for (int i = 0; i < num_clusters; i++) {
                prev_clusters[i].resize(cluster_sizes[i]);
                for (int k = 0; k < cluster_sizes[i]; k++) {
                    // Compute the group cluster center
                    int city_idx = groups[group_idx][cnt++]; // current city in this mini-cluster
                    prev_clusters[i][k] = city_idx;
                    prev_cluster_centers[i].first += problem.cities[city_idx].cx;
                    prev_cluster_centers[i].second += problem.cities[city_idx].cy;
                }

                // Normalize the cluster cetner
                prev_cluster_centers[i].first /= cluster_sizes[i];
                prev_cluster_centers[i].second /= cluster_sizes[i];
            }

            for (int iter = 0; iter < kmeans_iters; iter++) {
                vector<int> mini_cluster_indices = generate_indices(num_clusters);
                shuffle(mini_cluster_indices.begin(), mini_cluster_indices.end(), rng);

                // Indices of cities w.r.t this group that have been selected by mini-cluster already
                vector<bool> taken_cities(problem.group_sizes[group_idx]);

                vector<pair<ld, ld>> curr_cluster_centers(num_clusters);
                vector<vector<int>> curr_clusters = prev_clusters;

                for (int mini_cluster_idx : mini_cluster_indices) {
                    // Find the cluster_sizes[mini_cluster_idx] closest cities in this group to this city center
                    // Get all distances to non-taken cities in this group
                    vector<pair<ld, int>> distances;
                    for (int i = 0; i < problem.group_sizes[group_idx]; i++) {
                        if (taken_cities[i]) continue;
                        int city_idx = groups[group_idx][i];

                        distances.push_back({
                            dist(problem.cities[city_idx].cx, problem.cities[city_idx].cy, prev_cluster_centers[mini_cluster_idx].first, prev_cluster_centers[mini_cluster_idx].second), i
                        });
                    }

                    sort(distances.begin(), distances.end());
                    // Take the first cluster_sizes[mini_cluster_idx] closest cities, compute the new
                    for (int i = 0; i < cluster_sizes[mini_cluster_idx]; i++) {
                        int group_city_idx = distances[i].second;
                        taken_cities[group_city_idx] = true;

                        int city_idx = groups[group_idx][group_city_idx];
                        // Update cluster assignments
                        curr_clusters[mini_cluster_idx][i] = city_idx;

                        // Update cluster center
                        curr_cluster_centers[mini_cluster_idx].first += problem.cities[city_idx].cx;
                        curr_cluster_centers[mini_cluster_idx].second += problem.cities[city_idx].cy;
                    }

                    // Normalize the new cluster center
                    curr_cluster_centers[mini_cluster_idx].first /= cluster_sizes[mini_cluster_idx];
                    curr_cluster_centers[mini_cluster_idx].second /= cluster_sizes[mini_cluster_idx];
                }

                // Update prev assignments
                prev_cluster_centers = curr_cluster_centers;
                bool done_flag = false; // If no changes, just break
                // TODO: is it possible that given another random assignment of mini cluster indices, it could still improve it?
                if (curr_clusters == prev_clusters) {
                    done_flag = true;
                }
                prev_clusters = curr_clusters;

                if (done_flag) break;
            }

            clustered_groups[group_idx] = prev_clusters;

        }

        return clustered_groups;
    }


    void MST_with_clusters(vector<vector<int>> &clusters) {
        // Given K clusters being the city indices, find the K - 1 best edges that connect these clusters

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


/*
*

mar 30, 2025:
allowing to run for all kmeans iterations doesn't really do anything - seems like it converges very fast
still have very bad msts spanning the entire grid. issue with k-means?

iterate through cluster centers that are near the edges **first** to prevent really bad assignments of points near border
- issue: if it's already assigned to a really bad spot, could just be really bad, taking a lot of points near border, but the
group center could still be near enter

randomly initialize groups to be uniform? - still doesn't help

What about prioritizing groups w/ very high avg dist, also incorporating a term of either
- how far the center is from the border, or
- how far, on avg, the points are from the border.


next idea: use mst clusters for finding best connecting edges

overlapping queries? Somehow using these to find better edges?

*/
