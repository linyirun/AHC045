#include <bits/stdc++.h>
#define ll long long
#define ld long double
#define pii pair<int, int>
#define pll pair<ll, ll>
using namespace std;

// #define int ll
//const int MOD = 998244353;
const int MOD = 1000000007;
const int INF = 1e7;

// TODO: make this template to support lds?
// TODO: make this work for non-assignment problems too
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

    // ld shortest_path() {
    //     /*
    //      * Finds the shortest path, sends 1 unit of flow in this path.
    //      * Directly updates the edges on this path
    //      *
    //      * returns -1 if no path found, otherwise returns the total cost of this path
    //      */
    //
    //     vector<ld> dist(n, INF);
    //     vector<int> prev(n, -1); // Previous node that constructed this shortest path
    //
    //     dist[this->src_idx] = 0; // src node should have 0 dist
    //
    //     // Run Bellman-Ford
    //     for (int iter = 0; iter < n; iter++) {
    //         for (int i = 0; i < n; i++) {
    //             for (shared_ptr<Edge> &edge : adj[i]) {
    //                 // Test the forward edge from i->v
    //                 if (edge->from == i && edge->curr < edge->capacity) {
    //                     if (dist[i] + edge->cost < dist[edge->to]) {
    //                         dist[edge->to] = dist[i] + edge->cost;
    //                         prev[edge->to] = i;
    //                     }
    //                 }
    //
    //                 // Test the backward edge from v->i
    //                 if (edge->to == i && edge->curr > 0) {
    //                     if (dist[i] - edge->cost < dist[edge->from]) {
    //                         dist[edge->from] = dist[i] - edge->cost;
    //                         prev[edge->from] = i;
    //                     }
    //                 }
    //             }
    //         }
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

    // TODO: check potentials again
    ld shortest_path(vector<ld> &pi) {
    /*
     * Finds the shortest path using Dijkstra with potentials (Johnson's algorithm)
     * Sends 1 unit of flow along this path and updates edge flows and costs
     *
     * returns -1 if no path is found, otherwise returns total cost of the path
     */
        using T = pair<ld, int>; // (distance, node)
        priority_queue<T, vector<T>, greater<T>> pq;

        vector<ld> dist(n, INF);
        vector<int> prev(n, -1); // previous node
        vector<shared_ptr<Edge>> how(n, nullptr); // edge used to reach node

        dist[src_idx] = 0;
        pq.emplace(0, src_idx);

        while (!pq.empty()) {
            auto [d, u] = pq.top(); pq.pop();
            if (d > dist[u]) continue;

            for (auto &e : adj[u]) {
                // Forward edge
                if (e->from == u && e->curr < e->capacity) {
                    int v = e->to;
                    ld reduced_cost = e->cost + pi[u] - pi[v];
                    if (dist[u] + reduced_cost < dist[v]) {
                        dist[v] = dist[u] + reduced_cost;
                        prev[v] = u;
                        how[v] = e;
                        pq.emplace(dist[v], v);
                    }
                }
                // Backward edge
                if (e->to == u && e->curr > 0) {
                    int v = e->from;
                    ld reduced_cost = -e->cost + pi[u] - pi[v];
                    if (dist[u] + reduced_cost < dist[v]) {
                        dist[v] = dist[u] + reduced_cost;
                        prev[v] = u;
                        how[v] = e;
                        pq.emplace(dist[v], v);
                    }
                }
            }
        }

        if (dist[target_idx] >= INF) return -1;

        // Update potentials
        for (int i = 0; i < n; i++) {
            if (dist[i] < INF) pi[i] += dist[i];
        }

        // Update edges along the path
        ld path_cost = 0;
        int curr = target_idx;
        while (curr != src_idx) {
            auto e = how[curr];
            int u = prev[curr];
            if (e->from == u && e->to == curr) {
                e->curr++;
                path_cost += e->cost;
            } else {
                e->curr--;
                path_cost -= e->cost;
            }
            curr = u;
        }
        return path_cost;
    }



    ld min_cost_max_flow() {
        // Finds and returns the max flow
        // Updates adj, can reconstruct the residual graph from there
        ld total_cost = 0;
        ld cost = 0;
        int cnt = 0;
        vector<ld> pi(n, 0);
        while ((cost = shortest_path(pi)) != -1) {
            total_cost += cost;
            // cerr << "Done shortest_path iter " << cnt << '\n';
            cnt++;
        }

        return total_cost;
    }
};
int n;
vector<vector<shared_ptr<MCMF::Edge>>> adj;



int32_t main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> n;
    adj.resize(2 * n + 2);
    // src = 0, target = 2n + 1

    for (int i = 1; i <= n; i++) {
        for (int j = n + 1; j <= 2 * n; j++) {
            int cost;
            cin >> cost;
            // Make edge and add it to adj list
            auto edge = make_shared<MCMF::Edge>(i, j, 1, cost);
            adj[i].push_back(edge);
            adj[j].push_back(edge);
        }
    }

    // Add edge from source to every worker, and edge from every worker to sink
    for (int i = 1; i <= n; i++) {
        auto edge = make_shared<MCMF::Edge>(0, i, 1, 0);
        adj[0].push_back(edge);
        adj[i].push_back(edge);
    }
    for (int i = n + 1; i <= 2 * n; i++) {
        auto edge = make_shared<MCMF::Edge>(i, 2 * n + 1, 1, 0);
        adj[i].push_back(edge);
        adj[2 * n + 1].push_back(edge);
    }

    MCMF mcmf_obj(2 * n + 2, 0, 2 * n + 1, adj);
    int cost = round(mcmf_obj.min_cost_max_flow());

    // Reconstruct the assignments
    vector<int> assignments(n + 1);
    for (int i = 1; i <= n; i++) {
        for (shared_ptr<MCMF::Edge> &edge: adj[i]) {
            if (edge->curr > 0 && n + 1 <= edge->to && edge->to <= 2 * n) {
                assignments[i] = edge->to;
            }
        }
    }

    cout << cost << '\n';
    for (int i = 1; i <= n; i++) {
        cout << i << ' ' << assignments[i] - n << '\n';
    }


}
