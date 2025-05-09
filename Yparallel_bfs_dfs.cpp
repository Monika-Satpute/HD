#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

class Graph {
    int V; // Number of vertices
    vector<vector<int>> adj; // Adjacency list

public:
    Graph(int V) : V(V), adj(V) {}

    // Function to add an edge to the graph
    void addEdge(int v, int w) {
        adj[v].push_back(w);
        adj[w].push_back(v); // Undirected graph
    }

    // Function to print the tree structure
    void printTreeStructure() {
        cout << "\nTree Structure:\n";
        cout << "      0\n";
        cout << "    /   \\\n";
        cout << "   1     2\n";
        cout << "  / \\   / \\\n";
        cout << " 3   4 5   6\n\n";
    }

    // Parallel BFS implementation
    void parallelBFS(int startVertex) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[startVertex] = true;
        q.push(startVertex);

        cout << "Parallel BFS traversal: ";
        while (!q.empty()) {
            // Process the current level in parallel
            #pragma omp parallel
            {
                #pragma omp single
                {
                    int levelSize = q.size();
                    for (int i = 0; i < levelSize; i++) {
                        int v = q.front();
                        q.pop();
                        cout << v << " ";

                        // Process neighbors in parallel
                        #pragma omp task firstprivate(v)
                        {
                            for (int neighbor : adj[v]) {
                                #pragma omp critical
                                {
                                    if (!visited[neighbor]) {
                                        visited[neighbor] = true;
                                        q.push(neighbor);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Sequential BFS for comparison
    void sequentialBFS(int startVertex) {
        vector<bool> visited(V, false);
        queue<int> q;

        visited[startVertex] = true;
        q.push(startVertex);

        cout << "Sequential BFS traversal: ";
        while (!q.empty()) {
            int v = q.front();
            q.pop();
            cout << v << " ";

            for (int neighbor : adj[v]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }
    }

    // Parallel DFS implementation
    void parallelDFS(int startVertex) {
        vector<bool> visited(V, false);
        stack<int> s;

        s.push(startVertex);

        cout << "Parallel DFS traversal: ";
        while (!s.empty()) {
            int v;
            #pragma omp critical
            {
                v = s.top();
                s.pop();
            }

            if (!visited[v]) {
                #pragma omp critical
                {
                    visited[v] = true;
                    cout << v << " ";
                }

                // Process neighbors in parallel
                #pragma omp parallel for
                for (int i = 0; i < adj[v].size(); i++) {
                    int neighbor = adj[v][i];
                    #pragma omp critical
                    {
                        if (!visited[neighbor]) {
                            s.push(neighbor);
                        }
                    }
                }
            }
        }
    }

    // Sequential DFS for comparison
    void sequentialDFS(int startVertex) {
        vector<bool> visited(V, false);
        stack<int> s;

        s.push(startVertex);

        cout << "Sequential DFS traversal: ";
        while (!s.empty()) {
            int v = s.top();
            s.pop();

            if (!visited[v]) {
                visited[v] = true;
                cout << v << " ";

                for (int neighbor : adj[v]) {
                    if (!visited[neighbor]) {
                        s.push(neighbor);
                    }
                }
            }
        }
    }
};

int main() {
    // Create a tree structure (undirected graph)
    Graph g(7);
    g.addEdge(0, 1);  //      0
    g.addEdge(0, 2);  //    /   \
    g.addEdge(1, 3);  //   1     2
    g.addEdge(1, 4);  //  / \   / \
    g.addEdge(2, 5);  // 3   4 5   6
    g.addEdge(2, 6);

    // Print the tree structure
    g.printTreeStructure();

    // Test Parallel BFS
    auto start = high_resolution_clock::now();
    g.parallelBFS(0);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "\nParallel BFS took " << duration.count() << " microseconds.\n\n";

    // Test Sequential BFS
    start = high_resolution_clock::now();
    g.sequentialBFS(0);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "\nSequential BFS took " << duration.count() << " microseconds.\n\n";

    // Test Parallel DFS
    start = high_resolution_clock::now();
    g.parallelDFS(0);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "\nParallel DFS took " << duration.count() << " microseconds.\n\n";

    // Test Sequential DFS
    start = high_resolution_clock::now();
    g.sequentialDFS(0);
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    cout << "\nSequential DFS took " << duration.count() << " microseconds.\n";

    return 0;
}