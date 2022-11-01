#include <string>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <bitset>
#include <cstdint>
#include <sstream>
#include <iomanip>
#include <atomic>
#include <cmath>
#include <limits>
#include <omp.h>
#include <algorithm>

using namespace std;

//GCC DEBUG - set threads from command line during compilation using -D THREADS_COUNT=<VALUE>
#ifdef THREADS_COUNT
  #define THRDS THREADS_COUNT
#else
  #define THRDS omp_get_max_threads()
#endif

//Declarations
//Maximum N is currently 64, the program can be simly modified to use N>64 by changing MAX_N as well as the configuration type (for example to bitset<N>...)
typedef uint64_t configuration_type;
#define MAX_N 64

//ReadWrite Atomics (need to be written to by a single thread at a time)
//Best configuration globally
configuration_type best_configuration = 0;
//Best price, starts at max
double best_price = numeric_limits<double>::max();
//Counter for amount of recursive function calls
atomic<uint64_t> counter(0);

//Read Atomics (implicit, require read atomicity)
//N, K, A parameters from user input
int64_t n, k, a;
//Keeping graph on stack (as matrix)
double graph[MAX_N][MAX_N];
//Maximum recursive depth for spawning new tasks, value is adjusted dynamically in main()
int64_t max_depth = 5;

//Print graph as matrix
void print_problem() noexcept {	
	cout << "n: " << n << ", k: " << k << ", a: " << a << endl;
	for (int64_t i = 0; i < n; ++i) {
		for (int64_t j = 0; j < n; ++j) {
			cout << setw(5) << graph[i][j] << ' ';
		}
		cout << endl;
	}
}

//Majority of variables are kept local -> only one critical section needed (assuming read atomicity)
void solve(configuration_type configuration, double current_price, int64_t index, int64_t value, int64_t num_of_zeroes, int64_t num_of_ones) noexcept {
	//Frequent access to global memory causes major overhead (800% on my system) for both atomic<> and pragma omp atomic, counter only enabled in debug
	#ifdef DEBUG
		++counter;
	#endif
	//Using bitmap for configurations, places 1 (or 0 but it is default there) at the index, in this case, += is same as |=
	configuration += value << index;

	//If we put the index (vertex) into X (0), we have to add price of all edges that go to vertices of Y (1), this also sums non-existent edges but their price is set to 0 so it is ignored
	if (value == 0) {
		for (int64_t i = 0; i < index; ++i) {
			if ((configuration & (uint64_t(1) << i)) != 0) {
				current_price += graph[index][i];
			}
		}
	//Same as above except we add the vertex into Y (1) and sum all edges that go to vertices of X (0), .....
	} else {
		for (int64_t i = 0; i < index; ++i) {
			if ((configuration & (uint64_t(1) << i)) == 0) {
				current_price += graph[index][i];
			}
		}
	}

	//First cut-through - if the price is higher than max price, no need to continue
	//Overhead of critical section would probably be bigger than cut through missed due to data race
	if (current_price > best_price) {
		return;
	}
	
	//If we reached best price so far and the sizes of X and Y match, remember the configuration and price as best
	if (current_price < best_price && num_of_ones == n - a && num_of_zeroes == a) {
		//Double-checked locking for efficiency
		#pragma omp critical(BETTER_PRICE_FOUND)
		{
			if (current_price < best_price) {
				best_price = current_price;
				best_configuration = configuration;
			}
		}		
	}

	//Continue to next index, check if we are still in bounds
	if (++index < n) {
		//Second cut-through, if we need A zeroes and we already have more or equal of them, no need to do this branch
		if (num_of_zeroes < a) {
			//Spawning new tasks only at the start (creating pool)
			//Needs enough tasks to distribute the load properly
			//If there are too many, overhead will be bad
			#pragma omp task if (index <= max_depth)
			{
				solve(configuration, current_price, index, 0, num_of_zeroes + 1, num_of_ones);
			}
		}
		//Third cut-through, if we need N-A ones and we already have more or equal of them, no need to do this branch
		if (num_of_ones < n - a) {
			#pragma omp task if (index <= max_depth)
			{
				solve(configuration, current_price, index, 1, num_of_zeroes, num_of_ones + 1);
			}
		}
	}
}

void solve_start() noexcept {
	//Default uses # of cpu threads but can be set during compilation
    #pragma omp parallel num_threads(THRDS)
	{
		//Only one thread starts tasks
		#pragma omp single
		{
			//Always start left and right recursion branch as a separate task for faster parallel cut-through
			#pragma omp task
			{
				solve(0, 0, 0, 0, 1, 0);
			}
			#pragma omp task
			{
				solve(0, 0, 0, 1, 0, 1);
			}
		}
	}
    
	cout << "price: " << best_price << endl;
	cout << "configuration: ";
	for (int64_t i = n - 1; i >= 0; --i) {
		cout << ((best_configuration & (uint64_t(1) << i)) > 0);
	}
	cout << endl;
	
	#ifdef DEBUG
		cout << "function calls: " << counter << endl;
	#endif
}

int main(void) {
	cout << "threads used for computing: " << THRDS << endl;
	string tmp_line = "";

	//Load first line of file into memory
	{
		getline(cin, tmp_line);
		stringstream parser(tmp_line);
		parser >> n >> k >> a;		
	}

	//Load graph into matrix (fast loading into memory, then parsing)
	int i = 0, j = 0;
	double price = 0.0;
	while (getline(cin, tmp_line)) {
		stringstream parser(tmp_line);
		parser >> i >> j >> price;
		graph[i][j] = price;
		graph[j][i] = price;
	}
	
	//Calculate max depth as log2 of number of threads + constant
	//There will be approximately 2 + 2^max_depth+1 tasks created
	//For 2-3 threads, there should be roughly 66 tasks
	//For 4-7 threads, there should be roughly 130 tasks
	//For 8-15 threads, there should be roughly 258 tasks
	max_depth = log2(THRDS) + 4;
	//Solve the problem
	solve_start();

	return 0;
}