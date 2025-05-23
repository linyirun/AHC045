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
// const int INF = 1e7;


// VERSION: MCMF with initial dijkstra potential for clustering
// K-means for initial clustering with mst for subqueries
// mini-clustering within the groups too

// MCMF
class MCMF {
private:
    const ld INF = 1e18;
public:
    // MCMF for assignments (capacity = 1)
    struct Edge {
        int from, to; // u -> v
        int capacity; // total capacity;
        ld cost; // cost to send 1 unit of flow
        int curr; // curr number of flow

        Edge(int from, int to, int capacity, ld cost, int curr=0) : from(from), to(to), capacity(capacity), cost(cost), curr(curr) {};
    };

    vector<vector<shared_ptr<Edge>>> adj;
    int n; // number of nodes in the graph
    int src_idx, target_idx;

    MCMF(int n, int src_idx, int target_idx, vector<vector<shared_ptr<Edge>>> &adj) {
        /* Parameters:
         * n: number of nodes in the graph
         * src_idx: idx of the source node
         * target_idx: idx of the sink
         * adj: adjacency list of shared pointers to edges. An edge must be in both endpoints.
         */
        this->src_idx = src_idx;
        this->target_idx = target_idx;
        this->n = n;
        this->adj = adj;
    }

    // TODO: finish implementing this potential verison
    // ld shortest_path(vector<ld> &pi) {
    //     /*
    //      * Finds the shortest path, sends 1 unit of flow in this path.
    //      * Directly updates the edges on this path
    //      *
    //      * returns -1 if no path found, otherwise returns the total cost of this path
    //      */
    //
    //     // TODO: check stability using ld?
    //
    //     vector<ld> dist(n, INF);
    //     vector<int> prev(n, -1); // Previous node that constructed this shortest path
    //
    //     dist[this->src_idx] = 0; // src node should have 0 dist
    //
    //     // Given potentials pi, use Dijkstra to find the shortest path
    //     priority_queue<pair<ld, int>, vector<pair<ld, int>>, greater<pair<ld, int>>> pq;
    //     pq.push({0, this->src_idx});
    //
    //     while (!pq.empty()) {
    //         auto [curr_dist, node] = pq.top();
    //         pq.pop();
    //         if (curr_dist > dist[node]) continue;
    //
    //         for (shared_ptr<Edge> &edge : adj[node]) {
    //             // Forward edge
    //             if (edge->from == node && edge->curr < edge->capacity) {
    //                 ld new_dist = curr_dist + edge->cost + pi[edge->from] - pi[edge->to];
    //                 if (new_dist < dist[edge->to]) {
    //                     dist[edge->to] = new_dist;
    //                     prev[edge->to] = node;
    //                     pq.push({new_dist, edge->to});
    //                 }
    //             }
    //
    //             // Backward edge
    //             if (edge->to == node && edge->curr > 0) {
    //                 ld new_dist = curr_dist + (-edge->cost + pi[edge->from] - pi[edge->to]);
    //                 if (new_dist < dist[edge->to]) {
    //                     dist[edge->to] = new_dist;
    //                     prev[edge->to] = node;
    //                     pq.push({new_dist, edge->to});
    //                 }
    //             }
    //         }
    //     }
    //
    //     // Update the potentials
    //     for (int i = 0; i < n; i++) {
    //         pi[i] += dist[i];
    //     }
    //
    //     vector<int> nodes_in_path;
    //     if (dist[this->target_idx] >= INF) {
    //         return -1; // No path found
    //     }
    //
    //     ld cost = 0;
    //     // Reconstruct the shortest path, and update the edges
    //     int curr_node = target_idx;
    //     while (curr_node != this->src_idx) {
    //         int prev_node = prev[curr_node];
    //         // Find the edge
    //         shared_ptr<Edge> curr_edge;
    //         bool backward_edge = false;
    //         for (shared_ptr<Edge> &edge : adj[curr_node]) {
    //             if (edge->from == prev_node && edge->to == curr_node) {
    //                 curr_edge = edge;
    //                 break;
    //             }
    //             if (edge->from == curr_node && edge->to == prev_node) {
    //                 // This is a backward edge then, should subtract
    //                 backward_edge = true;
    //                 curr_edge = edge;
    //                 break;
    //             }
    //         }
    //         // Assert that we found a valid path
    //         assert(curr_edge);
    //
    //         if (backward_edge) {
    //             curr_edge->curr--;
    //             cost -= curr_edge->cost;
    //         } else {
    //             curr_edge->curr++;
    //             cost += curr_edge->cost;
    //         }
    //
    //         curr_node = prev_node;
    //     }
    //     return cost;
    // }

        ld shortest_path(vector<ld> &pi) {
        /*
         * Finds the shortest path, sends 1 unit of flow in this path.
         * Directly updates the edges on this path
         *
         * returns -1 if no path found, otherwise returns the total cost of this path
         */

        vector<ld> dist(n, INF);
        vector<int> prev(n, -1); // Previous node that constructed this shortest path

        dist[this->src_idx] = 0; // src node should have 0 dist

        priority_queue<pair<ld, int>, vector<pair<ld, int>>, greater<>> pq;
        vector<shared_ptr<Edge>> how(n); // edge we used to get here
        pq.push({0, this->src_idx});

        while (!pq.empty()) {
            auto [curr_dist, node] = pq.top();
            pq.pop();
            if (curr_dist > dist[node]) continue;

            for (shared_ptr<Edge> &edge : adj[node]) {
                // Forward edge
                if (edge->from == node && edge->curr < edge->capacity) {
                    ld new_dist = curr_dist + edge->cost + pi[node] - pi[edge->to];
                    if (new_dist < dist[edge->to]) {
                        dist[edge->to] = new_dist;
                        prev[edge->to] = node;
                        how[edge->to] = edge;
                        pq.push({new_dist, edge->to});
                    }
                }

                // Backward edge
                if (edge->to == node && edge->curr > 0) {
                    ld new_dist = curr_dist + (-edge->cost + pi[node] - pi[edge->from]);
                    if (new_dist < dist[edge->from]) {
                        dist[edge->from] = new_dist;
                        prev[edge->from] = node;
                        how[edge->from] = edge;
                        pq.push({new_dist, edge->from});
                    }
                }
            }
        }

        // Update the potentials
        for (int i = 0; i < n; i++) {
            // cerr << "dist[" << i << "]: " << dist[i] << '\n';
            if (dist[i] < INF - 1e-3) {
                pi[i] += dist[i];
            }
        }
        vector<int> nodes_in_path;
        if (dist[this->target_idx] >= INF) {
            return -1; // No path found
        }

        ld cost = 0;
        // Reconstruct the shortest path, and update the edges
        int curr_node = target_idx;
        while (curr_node != this->src_idx) {
            int prev_node = prev[curr_node];
            // Find the edge
            shared_ptr<Edge> curr_edge;
            bool backward_edge = false;
            // for (shared_ptr<Edge> &edge : adj[curr_node]) {
            //     if (edge->from == prev_node && edge->to == curr_node) {
            //         curr_edge = edge;
            //         break;
            //     }
            //     if (edge->from == curr_node && edge->to == prev_node) {
            //         // This is a backward edge then, should subtract
            //         backward_edge = true;
            //         curr_edge = edge;
            //         break;
            //     }
            // }
            curr_edge = how[curr_node];
            if (curr_edge->from == prev_node) backward_edge = false;
            else backward_edge = true;
            // Assert that we found a valid path
            assert(curr_edge);

            if (backward_edge) {
                curr_edge->curr--;
                cost -= curr_edge->cost;
            } else {
                curr_edge->curr++;
                cost += curr_edge->cost;
            }

            curr_node = prev_node;
        }
        return cost;
    }



    ld min_cost_max_flow() {
        // Finds and returns the max flow
        // Updates adj, can reconstruct the residual graph from there
        ld total_cost = 0;
        ld cost = 0;
        int cnt = 0;
        vector<ld> pi(n, 0);
        vector<int> prev(n, -1);

        // Run Bellman-Ford to calculate a potential

        pi[this->src_idx] = 0; // src node should have 0 dist

        // Run Bellman-Ford
        // for (int iter = 0; iter < n; iter++) {
        //     for (int i = 0; i < n; i++) {
        //         for (shared_ptr<Edge> &edge : adj[i]) {
        //             // Test the forward edge from i->v
        //             if (edge->from == i && edge->curr < edge->capacity) {
        //                 if (pi[i] + edge->cost < pi[edge->to]) {
        //                     pi[edge->to] = pi[i] + edge->cost;
        //                     prev[edge->to] = i;
        //                 }
        //             }
        //
        //             // Test the backward edge from v->i
        //             if (edge->to == i && edge->curr > 0) {
        //                 if (pi[i] - edge->cost < pi[edge->from]) {
        //                     pi[edge->from] = pi[i] - edge->cost;
        //                     prev[edge->from] = i;
        //                 }
        //             }
        //         }
        //     }
        // }

        while ((cost = shortest_path(pi)) != -1) {
            total_cost += cost;
            // cerr << "Done shortest_path iter " << cnt << '\n';
            cnt++;
        }

        return total_cost;
    }
};


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

class UnionFind {
public:
    vector<int> rep;
    vector<int> rank;

    UnionFind(int n) {
        rep.resize(n);
        rank.resize(n);
        for (int i = 0; i < n; i++) {
            rep[i] = i;
            rank[i] = 0;
        }
    }

    int get_rep(int x) {
        // Get the representative node of x.
        if (x == rep[x]) return x;
        return rep[x] = get_rep(rep[x]);
    }

    int same_set(int x, int y) {
        // Returns if x and y are in the same set.
        int rep_x = get_rep(x);
        int rep_y = get_rep(y);
        return rep_x == rep_y;
    }

    bool union_sets(int x, int y) {
        /* Union the sets containing x and y
         * Returns false if they are already in the same set.
         *
         * Performs union by rank
         */
        int rep_x = get_rep(x);
        int rep_y = get_rep(y);
        if (rep_x == rep_y) return false; // already in the same set

        if (rank[rep_x] < rank[rep_y]) {
            swap(rep_x, rep_y);
        } else if (rank[rep_x] == rank[rep_y]) {
            rank[rep_x]++;
        }
        rep[rep_y] = rep_x;

        return true;
    }

};




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

    static ld dist(ld x1, ld y1, ld x2, ld y2) {
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
                break;
            case 1:
                pt = {0, loc};
                break;
            case 2:
                pt = {loc, BORDER_SIZE};
                break;
            case 3:
                pt = {BORDER_SIZE, loc};
                break;
        }
        return pt;

    }


    vector<pii> cluster_kruskal(vector<vector<int>> &clusters) {
        /* Given clusters of size (M, cluster_size) of city_idxs
         * Returns a set of M - 1 edges that connect all clusters.
         */

        int M = clusters.size();
        vector<pair<ld, pii>> edges; // List of edges {dist, {cluster1, cluster2}}

        // connecting_nodes[i][j] = edge of min dist that connects these two clusters
        vector<vector<pii>> connecting_nodes(M, vector<pii>(M));

        for (int cluster1 = 0; cluster1 < M; cluster1++) {
            for (int cluster2 = cluster1 + 1; cluster2 < M; cluster2++) {
                // Find the min dist between these two clusters, along with the edge that does this
                pii best_edge = {0, 0};
                ld best_dist = LDBL_MAX;
                for (int city_idx1 : clusters[cluster1]) {
                    for (int city_idx2 : clusters[cluster2]) {
                        ld curr_dist = dist_center(city_idx1, city_idx2);
                        if (curr_dist < best_dist) {
                            best_dist = curr_dist;
                            best_edge = {city_idx1, city_idx2};
                        }
                    }
                }

                // Store this edge with its cost
                connecting_nodes[cluster1][cluster2] = connecting_nodes[cluster2][cluster1] = best_edge;

                // This is the best dist to connect cluster1 to cluster2
                edges.push_back({best_dist, {cluster1, cluster2}});
                // cerr << "Edge: " << cluster1 << ", " << cluster2 << '\n';
            }
        }

        // Perform Kruskals to find the MST
        UnionFind uf(M);
        sort(edges.begin(), edges.end());

        vector<pii> mst_edges;

        // cerr << "M = " << M << ", MST edges: \n";
        for (auto e : edges) {
            int u = e.second.first, v = e.second.second;
            if (uf.union_sets(u, v)) {
                mst_edges.push_back({u, v});
                cerr << e.second.first << ' ' << e.second.second << '\n';
            }

        }

        assert(mst_edges.size() == M - 1);
        // Get the actual edges
        vector<pii> result(M - 1);
        for (int i = 0; i < M - 1; i++) {
            result[i] = connecting_nodes[mst_edges[i].first][mst_edges[i].second];
        }
        return result;
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
        // groups = mcmf_clustering(2, groups);

        vector<vector<pii>> edges(problem.M);
        // Query, then save the edges of the MST
        // Query the next L

        // WITH MINI-CLUSTERING --------------------------------------------------------
        // vector<vector<vector<int>>> clustered_groups = miniClusteringKMeans(groups, 10);
        // cout << "Got here3\n";
        vector<vector<vector<int>>> clustered_groups = mcmf_mini_clustering(groups, 5);

        for (int i = 0; i < problem.M; i++) {
            int prev_group_idx = -1; // One representative node in the prev subset
            int num_clusters = clustered_groups[i].size();

            for (int curr_cluster_idx = 0; curr_cluster_idx < num_clusters; curr_cluster_idx++) {
                vector<int> to_query = clustered_groups[i][curr_cluster_idx]; // the set of cities in this mini-cluster
                // Query these and add it to edges
                vector<pii> ans = query(to_query);
                edges[i].insert(edges[i].end(), ans.begin(), ans.end());

                // if (prev_group_idx != -1) {
                //     // If there's a prev group, link this component to the prev
                //     // edges[i].push_back({prev_group_idx, groups[i][j]});
                //
                //     // TODO: implement clustering MST instead of this
                //
                //     pii best_edge = {prev_group_idx, clustered_groups[i][curr_cluster_idx][0]};
                //     ld best_dist = LDBL_MAX;
                //
                //     // Find the closest node in this cluster and any of the previous cluster, and join them
                //     for (int prev_cluster_idx = 0; prev_cluster_idx < curr_cluster_idx; prev_cluster_idx++) {
                //         for (int other_group_idx : clustered_groups[i][prev_cluster_idx]) {
                //             for (int this_group_idx : clustered_groups[i][curr_cluster_idx]) {
                //                 ld new_dist = dist_center(this_group_idx, other_group_idx);
                //                 if (new_dist < best_dist) {
                //                     best_dist = new_dist;
                //                     best_edge = {this_group_idx, other_group_idx};
                //                 }
                //             }
                //         }
                //     }
                //
                //     edges[i].push_back(best_edge);
                // }
                // prev_group_idx = clustered_groups[i][curr_cluster_idx][0];

            }

            // Run clustering MST alg to connect these subqueries
            vector<pii> connecting_mst_edges = cluster_kruskal(clustered_groups[i]);
            edges[i].insert(edges[i].end(), connecting_mst_edges.begin(), connecting_mst_edges.end());

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

    vector<vector<int>> mcmf_clustering(int num_iters, vector<vector<int>> &prev_groups) {
        /* Uses K-means with mcmf for better clustering
         * Does a max of num_iters soft K-means
         *
         *
         */

        // Initialize all group centers (just generate random points on the border)
        // uniform_real_distribution<ld> random_pt(0, 10000);
        // vector<pair<ld, ld>> prev_group_centers(problem.M);
        // for (int i = 0; i < problem.M; i++) {
            // Initialize to a random point on the border
            // prev_group_centers[i] = generate_border_point();
            // prev_group_centers[i] = {random_pt(rng), random_pt(rng)};
            // cerr << "group center " << i << ": " << prev_group_centers[i].first << ", " << prev_group_centers[i].second << '\n';
        // }

        // calculate the group centers
        vector<pair<ld, ld>> prev_group_centers(problem.M);
        for (int group_idx = 0; group_idx < problem.M; group_idx++) {
            ld center_x = 0, center_y = 0;
            for (int city_idx : prev_groups[group_idx]) {
                center_x += problem.cities[city_idx].cx;
                center_y += problem.cities[city_idx].cy;
            }
            prev_group_centers[group_idx] = {center_x, center_y};
        }

        // This is where we store the group assignments
        // vector<vector<int>> prev_groups(problem.M);
        // for (int group_idx = 0; group_idx < problem.M; group_idx++) {
        //     prev_groups[group_idx].resize(problem.group_sizes[group_idx]);
        // }

        // Main k-means loop --------------------------------------------------------
        for (int iter = 0; iter < num_iters; iter++) {
            // Run MCMF
            vector<int> city_indices = generate_indices(problem.N);
            vector<int> group_sizes = problem.group_sizes;
            vector<vector<int>> curr_groups = general_mcmf(city_indices, group_sizes, prev_group_centers, 50);
            bool done_flag = false;
            if (prev_groups == curr_groups) {
                cerr << "Done MCMF K-means after " << iter << " iterations.\n";
                done_flag = true;
            }
            // cerr << "Done MCMF iteration " << iter << '\n';

            // Recalculate the group centers
            vector<pair<ld, ld>> curr_group_centers(problem.M);
            for (int group_idx = 0; group_idx < problem.M; group_idx++) {
                ld center_x = 0, center_y = 0;
                for (int city_idx : curr_groups[group_idx]) {
                    center_x += problem.cities[city_idx].cx;
                    center_y += problem.cities[city_idx].cy;
                }
                curr_group_centers[group_idx] = {center_x, center_y};
            }

            prev_group_centers = curr_group_centers;
            prev_groups = curr_groups;
            if (done_flag) break;
        }

        return prev_groups;

    }

        vector<vector<vector<int>>> mcmf_mini_clustering(vector<vector<int>> &groups, int kmeans_iters) {
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
            vector<int> cluster_sizes(num_clusters, problem.L), actual_cluster_sizes(num_clusters);
            int total = problem.group_sizes[group_idx];
            for (int i = 0; i < num_clusters; i++) {
                actual_cluster_sizes[i] = min(problem.L, total);
                // using strict groups
                cluster_sizes[i] = actual_cluster_sizes[i];
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
                for (int k = 0; k < actual_cluster_sizes[i]; k++) {
                    // Compute the group cluster center
                    int city_idx = groups[group_idx][cnt++]; // current city in this mini-cluster
                    prev_clusters[i][k] = city_idx;
                    prev_cluster_centers[i].first += problem.cities[city_idx].cx;
                    prev_cluster_centers[i].second += problem.cities[city_idx].cy;
                }

                // Normalize the cluster center
                prev_cluster_centers[i].first /= cluster_sizes[i];
                prev_cluster_centers[i].second /= cluster_sizes[i];
            }


            // TODO: make this work for max number of L
            for (int iter = 0; iter < kmeans_iters; iter++) {
                vector<int> city_indices(problem.group_sizes[group_idx]);
                for (int i = 0; i < problem.group_sizes[group_idx]; i++) {
                    city_indices[i] = groups[group_idx][i];
                }

                vector<vector<int>> curr_clusters = general_mcmf(city_indices, cluster_sizes, prev_cluster_centers, 25);
                curr_clusters = convert_city_indices(curr_clusters, city_indices);

                bool done_flag = false;
                if (prev_clusters == curr_clusters) {
                    // cerr << "Done mini-clustering MCMF K-means after " << iter << " iterations.\n";
                    done_flag = true;
                }
                // cerr << "Done MCMF iteration " << iter << '\n';


                // Recalculate the group centers
                vector<pair<ld, ld>> curr_cluster_centers(num_clusters);
                for (int cluster_idx = 0; cluster_idx < num_clusters; cluster_idx++) {
                    ld center_x = 0, center_y = 0;
                    for (int city_idx : curr_clusters[cluster_idx]) {
                        center_x += problem.cities[city_idx].cx;
                        center_y += problem.cities[city_idx].cy;
                    }
                    curr_cluster_centers[cluster_idx] = {center_x, center_y};
                }

                prev_cluster_centers = curr_cluster_centers;
                prev_clusters = curr_clusters;
                if (done_flag) break;
                }

                clustered_groups[group_idx] = prev_clusters;

        }

        return clustered_groups;
    }


    vector<vector<int>> general_mcmf(vector<int> &city_indices, vector<int> &cluster_sizes, vector<pair<ld, ld>> &prev_cluster_centers, int mx_edges=-1, int depth=0) {
        /* Performs general mcmf for clustering
         * Performs only 1 loop
         *
         * groups: size N, indices of the cities we're clustering
         * cluster_sizes: size M, max size on each cluster.
         * - cluster_sizes[i] can be 0, we just don't consider this
         * mx_edges: max number of edges this flow network is allowed to have. If -1, no limit on edges.
         *
         * Returns:
         * N x (max cluster_sizes[i]) array of assignments
         * - if mx_edges != 1, some city_indices may not be assigned to a cluster.
         *
         * Note: returns the indices with respect to city_indices. need to turn it back to city_indices afterwards
         */

        // cerr << "Currently evaluating |city_indices| = " << city_indices.size() << ", |cluster_sizes| = " << cluster_sizes.size() << '\n';
        // cerr << "curr depth is " << depth << '\n';
        depth++;

        int num_cities = city_indices.size();
        int num_clusters = cluster_sizes.size();

        if (mx_edges == -1) mx_edges = num_clusters;

        int num_flow_nodes = num_cities + num_clusters + 2;

        int src_idx = 0, sink_idx = num_cities + num_clusters + 1;
        // Cluster assignments
        vector<vector<int>> clusters(num_clusters);
        for (int i = 0; i < num_clusters; i++) {
            clusters[i].resize(cluster_sizes[i]);
        }

        // Construct the graph for flows
        vector<vector<shared_ptr<MCMF::Edge>>> adj(num_flow_nodes);

        for (int city_idx = 0; city_idx < num_cities; city_idx++) {
            // Calculate the distances to all the clusters
            vector<pair<ld, int>> dist_to_cluster_center(num_clusters);
            for (int cluster_idx = 0; cluster_idx < num_clusters; cluster_idx++) {
                // Cost of this edge [city_idx][group_idx] will be the distance between this city and the city center
                City *city_ptr = &problem.cities[city_indices[city_idx]];
                ld dist_to_center = dist(city_ptr->cx, city_ptr->cy, prev_cluster_centers[cluster_idx].first, prev_cluster_centers[cluster_idx].second);
                dist_to_cluster_center[cluster_idx] = {dist_to_center, cluster_idx};
            }

            sort(dist_to_cluster_center.begin(), dist_to_cluster_center.end());
            // Only take the mx_edges closest clusters
            for (int idx = 0; idx < min((int) dist_to_cluster_center.size(), mx_edges); idx++) {
                int cluster_idx = dist_to_cluster_center[idx].second;
                ld dist_to_center = dist_to_cluster_center[idx].first;
                int flow_city_idx = 1 + city_idx;
                int flow_group_idx = 1 + num_cities + cluster_idx;
                shared_ptr<MCMF::Edge> edge = make_shared<MCMF::Edge>(flow_city_idx, flow_group_idx, 1, dist_to_center);
                adj[flow_city_idx].push_back(edge);
                adj[flow_group_idx].push_back(edge);
            }
        }

        for (int city_idx = 0; city_idx < num_cities; city_idx++) {
            int flow_city_idx = 1 + city_idx;
            shared_ptr<MCMF::Edge> edge = make_shared<MCMF::Edge>(src_idx, flow_city_idx, 1, 0);
            adj[src_idx].push_back(edge);
            adj[flow_city_idx].push_back(edge);
        }
        for (int group_idx = 0; group_idx < num_clusters; group_idx++) {
            int flow_group_idx = 1 + num_cities + group_idx;
            // Make this edge with capacity group_size
            shared_ptr<MCMF::Edge> edge = make_shared<MCMF::Edge>(flow_group_idx, sink_idx, cluster_sizes[group_idx], 0);
            adj[flow_group_idx].push_back(edge);
            adj[sink_idx].push_back(edge);
        }

        // Setup and perform MCMF
        MCMF mcmf_obj(num_flow_nodes, src_idx, sink_idx, adj);
        mcmf_obj.min_cost_max_flow();

        // Get the assignments of cities to groups
        vector<int> city_to_group_assignments(num_cities, -1);
        for (int city_idx = 0; city_idx < num_cities; city_idx++) {
            int flow_city_idx = 1 + city_idx;
            for (shared_ptr<MCMF::Edge> &edge : adj[flow_city_idx]) {
                // If this node is the beginning of edge and there's flow, then this is assigned
                if (edge->from == flow_city_idx && edge->curr > 0) {
                    int assigned_group_idx = edge->to - num_cities - 1;
                    city_to_group_assignments[city_idx] = assigned_group_idx;
                    break;
                }
            }
        }

        // Construct the groups again
        // These curr_group indices are with respect to the current indices.
        vector<int> unassigned_cities;
        vector<vector<int>> curr_groups(num_clusters);
        for (int city_idx = 0; city_idx < num_cities; city_idx++) {
            if (city_to_group_assignments[city_idx] != -1) {
                curr_groups[city_to_group_assignments[city_idx]].push_back(city_idx);
            } else {
                unassigned_cities.push_back(city_idx);
            }
        }

        // TODO: if there's any unassigned cities, then re-run this algorithm, only on those unassigned
        // Create the new group sizes based off of any group that has space

        // Maps ind to new cluster ind
        vector<int> new_cluster_sizes;
        vector<int> ind_to_cluster_ind;
        // Cluster centers of the available groups
        vector<pair<ld, ld>> cluster_centers_available;

        // actual city indexes of the city_idx
        vector<int> actual_unassigned_cities_idx = convert_city_indices(unassigned_cities, city_indices);

        for (int cluster_idx = 0; cluster_idx < num_clusters; cluster_idx++) {
            int sz = cluster_sizes[cluster_idx] - curr_groups[cluster_idx].size();
            if (sz > 0) {
                ind_to_cluster_ind.push_back(cluster_idx);
                new_cluster_sizes.push_back(sz);
                cluster_centers_available.push_back(prev_cluster_centers[cluster_idx]);
            }
        }

        if (!unassigned_cities.empty()) {
            vector<vector<int>> new_assignments = general_mcmf(actual_unassigned_cities_idx, new_cluster_sizes, prev_cluster_centers, mx_edges, depth);
            // Convert it back and put it into the valid group
            for (int new_cluster_idx = 0; new_cluster_idx < new_cluster_sizes.size(); new_cluster_idx++) {
                int actual_cluster_idx = ind_to_cluster_ind[new_cluster_idx];
                // Convert the city index into the proper ones, with respect to this city's indices
                vector<int> this_city_inds = convert_city_indices(new_assignments[new_cluster_idx], unassigned_cities);
                for (int ind : this_city_inds) {
                    curr_groups[actual_cluster_idx].push_back(ind);
                }
            }
        }

        return curr_groups;
    }

    vector<vector<int>> convert_city_indices(vector<vector<int>> &cluster, vector<int> &city_indices) {
        // Turns clusters that have indices w.r.t city_indices into the actual city indices
        vector<vector<int>> actual_indices = cluster;
        for (int i = 0; i < cluster.size(); i++) {
            for (int j = 0; j < cluster[i].size(); j++) {
                actual_indices[i][j] = city_indices[cluster[i][j]];
            }
        }
        return actual_indices;
    }
    vector<int> convert_city_indices(vector<int> &cluster, vector<int> &city_indices) {
        // Turns clusters that have indices w.r.t city_indices into the actual city indices
        vector<int> actual_indices = cluster;
        for (int i = 0; i < cluster.size(); i++) {
            actual_indices[i] = city_indices[cluster[i]];
        }
        return actual_indices;
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

        // Get the max group size
        int max_group_size = 0;
        for (int group_size : problem.group_sizes) {
            max_group_size = max(group_size, max_group_size);
        }


        // 2) Main K-means loop ----------------------------------------------------------
        for (int iter = 0; iter < num_iters; iter++) {
            vector<bool> taken_cities(problem.N); // cities that have already been selected
            vector<vector<int>> curr_groups = prev_groups; // The current group assignments
            vector<pair<ld, ld>> curr_group_centers(problem.M);

            // Loop over the group indexes in random order, to not give preference to earliest groups
            vector<int> group_indices = generate_indices(problem.M);
            shuffle(group_indices.begin(), group_indices.end(), rng);

            // Get distance of its mean to its nearest border
            vector<ld> dist_to_border(problem.M);
            for (int i = 0; i < problem.M; i++) {
                pii center = prev_group_centers[i];
                dist_to_border[i] = min({center.first, center.second, 10000 - center.first, 10000 - center.second});
            }
            //
            // sort(group_indices.begin(), group_indices.end(), [&dist_to_border](const int i1, const int i2) {
            //     return dist_to_border[i1] < dist_to_border[i2];
            // });

            // Calculate, for each cluster, average distance of a cluster to its border
            vector<ld> avg_dist_to_center(problem.M);
            for (int i = 0; i < problem.M; i++) {
                pii center = prev_group_centers[i];
                for (int j = 0; j < problem.group_sizes[i]; j++) {
                    int city_idx = curr_groups[i][j];
                    avg_dist_to_center[i] += dist(problem.cities[city_idx].cx, problem.cities[city_idx].cy, center.first, center.second);
                }
                avg_dist_to_center[i] = sqrt(avg_dist_to_center[i]);
                avg_dist_to_center[i] /= problem.group_sizes[i];
            }

            // For each cluster, count the proportion of points that are near the border
            vector<int> num_near_border(problem.M);
            int BORDER_THRESHOLD = 1000;
            for (int group_idx = 0; group_idx < problem.M; group_idx++) {
                int cnt_near = 0;
                for (int city_idx : prev_groups[group_idx]) {
                    int x = problem.cities[city_idx].cx;
                    int y = problem.cities[city_idx].cy;
                    int curr_dist = min({x, y, 10000 - x, 10000 - y});
                    if (curr_dist <= BORDER_THRESHOLD) {
                        cnt_near++;
                    }
                }
                num_near_border[group_idx] = cnt_near;

            }

            // Get the "score" of each cluster based on how much priority it should get. lower is better
            vector<ld> score(problem.M);
#ifdef DEBUG
            cerr << "kmeans iteration " << iter << '\n';
#endif
            for (int i = 0; i < problem.M; i++) {
                // cerr << "got here " << i << "\n";
                // cerr << "problem M: " << problem.M << '\n';
                // score[i] = -10 * avg_dist_to_center[i] - problem.group_sizes[i] - 100 * exp(-dist_to_border[i] / 100);
                score[i] = -problem.group_sizes[i] - avg_dist_to_center[i] - 100 * ((ld) num_near_border[i] / problem.group_sizes[i]);
                // score[i] = -100 * ((ld) num_near_border[i]);

#ifdef DEBUG
                cerr << "score[" << i << "]: " << score[i] << '\n';
                // cerr << "avg dist: " << -3 * avg_dist_to_center[i] << ", " <<  "group size: " << problem.group_sizes[i] << ", dist to border: " << 100 * exp(-dist_to_border[i] / 100) << '\n';
                cerr << "group_size: " << problem.group_sizes[i] << ", avg_dist_to_center: " << 3 * avg_dist_to_center[i] << ", proportion near border: " << 100 * ((ld) num_near_border[i] / problem.group_sizes[i]) << '\n';
#endif
            }

            sort(group_indices.begin(), group_indices.end(), [&score](const int i1, const int i2) {
                return score[i1] < score[i2];
            });
#ifdef DEBUG
            cerr << "order of groups:\n";
            for (int i = 0; i < problem.M; i++) {
                cerr << group_indices[i] << ' ';
            }
            cerr << '\n';
#endif



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
        //         problem.cities[city_idx].group_idx_j = i;
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
        //             if (dist1_to_group2 < 0.6 * dist1_to_group1 && dist2_to_group1 < 0.6 * dist2_to_group2) {
        //                 // Perform the swap
        //                 swapped = true;
        //                 int group1_idx_j = problem.cities[city_idx1].group_idx_j, group2_idx_j = problem.cities[city_idx2].group_idx_j;
        //
        //                 swap(prev_groups[group1][group1_idx_j], prev_groups[group2][group2_idx_j]);
        //
        //                 swap(problem.cities[city_idx1].group_idx_j, problem.cities[city_idx2].group_idx_j);
        //                 swap(problem.cities[city_idx1].group_idx, problem.cities[city_idx2].group_idx);
        //
        //                 // Update the new centers
        //                 prev_group_centers[group1] = prev_group_centers[group2] = {0, 0};
        //                 for (int city_idx : prev_groups[group1]) {
        //                     prev_group_centers[group1].first += problem.cities[city_idx].cx;
        //                     prev_group_centers[group1].second += problem.cities[city_idx].cy;
        //                 }
        //                 prev_group_centers[group1].first /= problem.group_sizes[group1];
        //                 prev_group_centers[group1].second /= problem.group_sizes[group1];
        //
        //                 for (int city_idx : prev_groups[group2]) {
        //                     prev_group_centers[group2].first += problem.cities[city_idx].cx;
        //                     prev_group_centers[group2].second += problem.cities[city_idx].cy;
        //                 }
        //                 prev_group_centers[group2].first /= problem.group_sizes[group2];
        //                 prev_group_centers[group2].second /= problem.group_sizes[group2];
        //
        //
        //
        //
        //                 cerr << "swapped city indices " << city_idx1 << " " << city_idx2 << ", from group " << group1 << " to " << group2 << '\n';
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
        cin >> problem.cities[i].lx >> problem.cities[i].rx >> problem.cities[i].ly >> problem.cities[i].ry;
        // calculate the centers
        problem.cities[i].cx = (ld) (problem.cities[i].lx + problem.cities[i].rx) / 2;
        problem.cities[i].cy = (ld) (problem.cities[i].ly + problem.cities[i].ry) / 2;

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

i think we need to use number of points close to border, with a measure of how close. current approach does well but still
has very long edges


next idea: use mst clusters for finding best connecting edges

overlapping queries? Somehow using these to find better edges?


apr 1st:
implement MCMF: done

apr 2:
add optimization for mcmf, make only edges for each city to 10-15 nearest cities



6 1 3 3 500
6
1375 1648 351 624
1773 1900 3660 3787
2922 3231 558 867
5358 5640 8585 8867
3218 3684 3330 3796
3218 3684 3330 3796


run k-means first, then start with mcmf?

*/
