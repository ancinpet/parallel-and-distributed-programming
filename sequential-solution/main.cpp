#include <string>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <bitset>
#include <cstdint>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <limits>

using namespace std;

//Declarations
//Maximum N is currently 64, the program can be simly modified to use N>64 by changing MAX_N as well as the configuration type (for example to bitset<N>...)
typedef uint64_t configuration_type;
#define MAX_N 64

//Best configuration globally
configuration_type best_configuration = 0;
//Best price, starts at max
double best_price = std::numeric_limits<double>::max();
//Keeping graph on stack (as matrix)
double graph[MAX_N][MAX_N];
//N, K, A parameters from user input
int64_t n, k, a;
//Counter for amount of recursive function calls
uint64_t counter = 0;

//Prints the graph as matrix
void print_problem() noexcept {	
	cout << "n: " << n << ", k: " << k << ", a: " << a << endl;
	for (int64_t i = 0; i < n; ++i) {
		for (int64_t j = 0; j < n; ++j) {
			cout << setw(5) << graph[i][j] << ' ';
		}
		cout << endl;
	}
}

//Recursive function to solve the problem, only index needs to be copied, other items could be kept in array for more effectivity
void solve(configuration_type configuration, double current_price, int64_t index, int64_t value, int64_t num_of_zeroes, int64_t num_of_ones) noexcept {
	//Number of function calls
	++counter;
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
	if (current_price > best_price) {
		return;
	}

	//If we reached best price so far and the sizes of X and Y match, remember the configuration and price as best
	if (current_price < best_price && num_of_ones == n - a && num_of_zeroes == a) {
		best_price = current_price;
		best_configuration = configuration;
	}

	//Continue to next index, check if we are still in bounds
	if (++index < n) {
		//Second cut-through, if we need A zeroes and we already have more or equal of them, no need to do this branch
		if (num_of_zeroes < a) {
			solve(configuration, current_price, index, 0, num_of_zeroes + 1, num_of_ones);
		}
		//Third cut-through, if we need N-A ones and we already have more or equal of them, no need to do this branch
		if (num_of_ones < n - a) {
			solve(configuration, current_price, index, 1, num_of_zeroes, num_of_ones + 1);
		}
	}
}

void solve_start() noexcept {
	//Starts the solving for two branches, starting with configuration 00000.....000000, price 0, index 0, trying to put 0 into the index in first call, 1 in the second call
	solve(0, 0, 0, 0, 1, 0);
	solve(0, 0, 0, 1, 0, 1);
	//After we are done, print the solution
	cout << "price: " << best_price << endl;
	cout << "configuration: " << bitset<64>(best_configuration) << endl;
	cout << "calls: " << counter << endl;
}

int main(void) {
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
	
	//Solve the problem
	solve_start();

	return 0;
}