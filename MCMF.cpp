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
class MCMF {
public:
    // MCMF for assignments (capacity = 1)
    struct Edge {
        int from, to; // u -> v
        int capacity; // total capacity;
        int cost; // cost to send 1 unit of flow
        int curr; // curr number of flow

        Edge(int from, int to, int capacity, int cost, int curr=0) : from(from), to(to), capacity(capacity), cost(cost), curr(curr) {};
    };

    vector<vector<shared_ptr<Edge>>> adj;
    int n; // number of nodes in the graph
    int src_idx, target_idx;

    MCMF(int n, int src_idx, int target_idx, vector<vector<shared_ptr<Edge>>> &adj) {
        this->src_idx = src_idx;
        this->target_idx = target_idx;
        this->n = n;
        this->adj = adj;
    }

    int shortest_path() {
        /*
         * Finds the shortest path, sends 1 unit of flow in this path.
         * Directly updates the edges on this path
         *
         * returns -1 if no path found, otherwise returns the total cost of this path
         */

        vector<int> dist(n, INF);
        vector<int> prev(n, -1); // Previous node that constructed this shortest path

        dist[this->src_idx] = 0; // src node should have 0 dist

        // Run Bellman-Ford
        for (int iter = 0; iter < n; iter++) {
            for (int i = 0; i < n; i++) {
                for (shared_ptr<Edge> &edge : adj[i]) {
                    // Test the forward edge from i->v
                    if (edge->from == i && edge->curr < edge->capacity) {
                        if (dist[i] + edge->cost < dist[edge->to]) {
                            dist[edge->to] = dist[i] + edge->cost;
                            prev[edge->to] = i;
                        }
                    }

                    // Test the backward edge from v->i
                    if (edge->to == i && edge->curr > 0) {
                        if (dist[i] - edge->cost < dist[edge->from]) {
                            dist[edge->from] = dist[i] - edge->cost;
                            prev[edge->from] = i;
                        }
                    }
                }
            }
        }

        vector<int> nodes_in_path;
        if (dist[this->target_idx] >= INF) {
            return -1; // No path found
        }

        int cost = 0;
        // Reconstruct the shortest path, and update the edges
        int curr_node = target_idx;
        while (curr_node != this->src_idx) {
            int prev_node = prev[curr_node];
            // Find the edge
            shared_ptr<Edge> curr_edge;
            bool backward_edge = false;
            for (shared_ptr<Edge> &edge : adj[curr_node]) {
                if (edge->from == prev_node && edge->to == curr_node) {
                    curr_edge = edge;
                    break;
                }
                if (edge->from == curr_node && edge->to == prev_node) {
                    // This is a backward edge then, should subtract
                    backward_edge = true;
                    curr_edge = edge;
                    break;
                }
            }
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



    int min_cost_max_flow() {
        // Finds and returns the max flow
        // Updates adj, can reconstruct the residual graph from there
        int total_cost = 0;
        int cost = 0;
        while ((cost = shortest_path()) != -1) {
            total_cost += cost;
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
    int cost = mcmf_obj.min_cost_max_flow();

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
