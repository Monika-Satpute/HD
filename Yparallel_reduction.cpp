#include <iostream>
#include <vector>
#include <climits>  // For INT_MIN and INT_MAX
#include <omp.h>    // OpenMP header

int main() {
    std::vector<int> data = {10, 20, 5, 3, 7, 15, 8, 25, 13, 17};
    int n = data.size();

    int min_val = INT_MAX;
    int max_val = INT_MIN;
    int sum = 0;

    // Parallel region with reduction
    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val) reduction(+:sum)
    for (int i = 0; i < n; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
        sum += data[i];
    }

    double average = static_cast<double>(sum) / n;

    // Output
    std::cout << "Minimum: " << min_val << std::endl;
    std::cout << "Maximum: " << max_val << std::endl;
    std::cout << "Sum: " << sum << std::endl;
    std::cout << "Average: " << average << std::endl;

    return 0;
}
