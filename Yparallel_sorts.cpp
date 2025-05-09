#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <algorithm>

using namespace std;
using namespace std::chrono;

// Verify if array is sorted
bool is_sorted(int arr[], int size) {
    for(int i = 0; i < size-1; i++) {
        if(arr[i] > arr[i+1]) return false;
    }
    return true;
}

void sequential_bubble_sort(int arr[], int size) {
    int* array = new int[size];
    copy(arr, arr+size, array);

    auto start = high_resolution_clock::now();
    for(int i = 0; i < size - 1; i++) {
        bool swapped = false;
        for(int j = 0; j < size - i - 1; j++) {
            if(array[j] > array[j+1]) {
                swap(array[j], array[j+1]);
                swapped = true;
            }
        }
        if(!swapped) break;
    }
    auto end = high_resolution_clock::now();
    
    cout << "Sequential Bubble Sort:\n";
    cout << "  Time: " << duration_cast<microseconds>(end-start).count() << " μs";
    cout << " | Sorted: " << (is_sorted(array, size) ? "Yes" : "No") << endl;
    delete[] array;
}

void parallel_bubble_sort(int arr[], int size) {
    int* array = new int[size];
    copy(arr, arr+size, array);

    auto start = high_resolution_clock::now();
    bool swapped = true;
    for(int k = 0; k < size && swapped; k++) {
        swapped = false;
        if(k % 2 == 0) {
            #pragma omp parallel for shared(array, swapped)
            for(int i = 1; i < size - 1; i += 2) {
                if(array[i] > array[i+1]) {
                    swap(array[i], array[i+1]);
                    #pragma omp critical
                    swapped = true;
                }
            }
        }
        else {
            #pragma omp parallel for shared(array, swapped)
            for(int i = 0; i < size - 1; i += 2) {
                if(array[i] > array[i+1]) {
                    swap(array[i], array[i+1]);
                    #pragma omp critical
                    swapped = true;
                }
            }
        }
    }
    auto end = high_resolution_clock::now();
    
    cout << "Parallel Bubble Sort:\n";
    cout << "  Time: " << duration_cast<microseconds>(end-start).count() << " μs";
    cout << " | Sorted: " << (is_sorted(array, size) ? "Yes" : "No") << endl;
    delete[] array;
}

void merge(int array[], int low, int mid, int high) {
    int* temp = new int[high - low + 1];
    int i = low, j = mid + 1, k = 0;
    
    while(i <= mid && j <= high) {
        if(array[i] <= array[j]) temp[k++] = array[i++];
        else temp[k++] = array[j++];
    }
    
    while(i <= mid) temp[k++] = array[i++];
    while(j <= high) temp[k++] = array[j++];
    
    for(i = low, k = 0; i <= high; i++, k++) {
        array[i] = temp[k];
    }
    
    delete[] temp;
}

void mergesort(int array[], int low, int high) {
    if(low < high) {
        int mid = low + (high - low) / 2;
        mergesort(array, low, mid);
        mergesort(array, mid+1, high);
        merge(array, low, mid, high);
    }
}

void perform_merge_sort(int arr[], int size) {
    int* array = new int[size];
    copy(arr, arr+size, array);

    auto start = high_resolution_clock::now();
    mergesort(array, 0, size-1);
    auto end = high_resolution_clock::now();
    
    cout << "Sequential Merge Sort:\n";
    cout << "  Time: " << duration_cast<microseconds>(end-start).count() << " μs";
    cout << " | Sorted: " << (is_sorted(array, size) ? "Yes" : "No") << endl;
    delete[] array;
}

void p_mergesort(int array[], int low, int high) {
    if(low < high) {
        int mid = low + (high - low) / 2;
        #pragma omp parallel sections
        {
            #pragma omp section
            p_mergesort(array, low, mid);
            #pragma omp section
            p_mergesort(array, mid+1, high);
        }
        merge(array, low, mid, high);
    }
}

void perform_p_merge_sort(int arr[], int size) {
    int* array = new int[size];
    copy(arr, arr+size, array);

    auto start = high_resolution_clock::now();
    p_mergesort(array, 0, size-1);
    auto end = high_resolution_clock::now();
    
    cout << "Parallel Merge Sort:\n";
    cout << "  Time: " << duration_cast<microseconds>(end-start).count() << " μs";
    cout << " | Sorted: " << (is_sorted(array, size) ? "Yes" : "No") << endl;
    delete[] array;
}

int main() {
    srand(time(0));
    
    int sizes[] = {1000, 5000, 10000}; // Test with larger sizes
    int MAX = 10000;
    
    for(int size : sizes) {
        int* array = new int[size];
        for(int i = 0; i < size; i++) {
            array[i] = rand() % MAX;
        }
        
        cout << "\nSorting " << size << " elements:\n";
        sequential_bubble_sort(array, size);
        parallel_bubble_sort(array, size);
        perform_merge_sort(array, size);
        perform_p_merge_sort(array, size);
        
        delete[] array;
    }
    
    return 0;
}