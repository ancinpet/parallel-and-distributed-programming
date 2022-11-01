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

#define THRDS omp_get_max_threads()

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
//Maximum states for parallel for, value is adjusted dynamically in main()
uint64_t enough_states = 500;

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

//Print configuration as bitmap
void print_configuration(configuration_type configuration, int64_t size) noexcept {
	for (int64_t i = size - 1; i >= 0; --i) {
		cout << ((configuration & (uint64_t(1) << i)) > 0);
	}
	cout << endl;
}

//Fast Queue that can be iterated over
//Made specifically for this task, cannot be used as regular queue
class IterativeQueue {
public:
	typedef pair<configuration_type, int64_t> value_type;
	typedef value_type * iterator;
	typedef value_type const* const_iterator;
	typedef value_type & reference;
	typedef value_type const& const_reference;

	IterativeQueue(uint64_t size) noexcept: m_start(0), m_queue() {
		m_queue.reserve(1 + size * 2);
	}
	void push(value_type && item) noexcept {
		m_queue.emplace_back(item);
	}
	value_type pop() noexcept {
		return m_queue[m_start++];
	}
	iterator begin() noexcept {
        return &m_queue[m_start];
    }
    const_iterator begin() const noexcept {
        return &m_queue[m_start];
    }
    iterator end() noexcept {
        return &m_queue[m_queue.size()];
    }
    const_iterator end() const noexcept {
        return &m_queue[m_queue.size()];
    }
	const_reference at(uint64_t pos) const noexcept {
		return m_queue[m_start + pos];
	}
	uint64_t size() const noexcept {
		return m_queue.size() - m_start;
	}
private:
	uint64_t m_start;
	vector<value_type> m_queue;
};

//Function taken from previous homework (task parallelism) -> used for solving remainder of the problem
void solve_rest(configuration_type configuration, double current_price, int64_t index, int64_t value, int64_t num_of_zeroes, int64_t num_of_ones) noexcept {
	#ifdef DEBUG
		++counter;
	#endif
	configuration += value << index;

	if (value == 0) {
		for (int64_t i = 0; i < index; ++i) {
			if ((configuration & (uint64_t(1) << i)) != 0) {
				current_price += graph[index][i];
			}
		}
	} else {
		for (int64_t i = 0; i < index; ++i) {
			if ((configuration & (uint64_t(1) << i)) == 0) {
				current_price += graph[index][i];
			}
		}
	}

	if (current_price > best_price) {
		return;
	}
	
	if (current_price < best_price && num_of_ones == n - a && num_of_zeroes == a) {
		#pragma omp critical(BETTER_PRICE_FOUND)
		{
			if (current_price < best_price) {
				best_price = current_price;
				best_configuration = configuration;
			}
		}		
	}

	if (++index < n) {
		if (num_of_zeroes < a) {
			solve_rest(configuration, current_price, index, 0, num_of_zeroes + 1, num_of_ones);
		}
		if (num_of_ones < n - a) {
			solve_rest(configuration, current_price, index, 1, num_of_zeroes, num_of_ones + 1);
		}
	}
}

//Before we can call function that solves the remainder of the problem we have to calculate current price, number of zeroes and ones for faster solving
void solve_seq(const pair<configuration_type, int64_t> & subproblem) noexcept {
	double current_price = 0;
	configuration_type configuration = subproblem.first;
	int64_t index = subproblem.second;
	int64_t num_of_zeroes = 0, num_of_ones = 0;

	//Calculates the price of the current configuration and amount of zeroes and ones in the configuration
	for (int64_t i = 0; i <= index; ++i) {
		//Vertex at position i is in set Y
		bool left = (configuration & (uint64_t(1) << i)) > 0;
		if (left) {
			++num_of_ones;
		} else {
			++num_of_zeroes;
		}

		for (int64_t j = 0; j < i; ++j) {
			//Vertex at position j is in set Y
			bool right = (configuration & (uint64_t(1) << j)) > 0;
			//If i and j are in different vertex sets, add price
			if (left != right) {
				current_price += graph[i][j];
			}
		}
	}

	//Branch&Bounds by price
	if (current_price > best_price) {
		return;
	}

	if (++index < n) {
		//Branch&Bounds by #zeroes
		if (num_of_zeroes < a) {
			solve_rest(configuration, current_price, index, 0, num_of_zeroes + 1, num_of_ones);
		}
		//Branch&Bounds by #ones
		if (num_of_ones < n - a) {
			solve_rest(configuration, current_price, index, 1, num_of_zeroes, num_of_ones + 1);
		}
	}
}

//Each loop pops one configuration and adds two more
//Example: pop X -> push 0X, push 1X where X is some configuration f.e. 101001
void generate_enough_states(IterativeQueue & queue) noexcept {
	int64_t tasks_created = 0;
	int64_t max_overhead = exp2(n * 3 / 4);
	//Create enough states, also check for bad settings that could create problems
	while (queue.size() <= enough_states && ++tasks_created < max_overhead) {
		//BFS -> pop configuration
		auto left = queue.pop();
		//Create a copy of it, configuration is of type <configuration_type, index(size)>
		auto right = left;
		//Add 0 to the left (only increment index)
		++left.second;
		//Add 1 to the left (and increment index)
		right.first += 1 << ++right.second;

		//Add two new configurations to the end of the queue
		queue.push(move(left));
		queue.push(move(right));
	}
}

void solve_start() noexcept {
	//Create queue that uses array underneath
	IterativeQueue queue(enough_states);
	//Push configuration that doesn't exist but works for generating others
	queue.push(move(make_pair(0, -1)));
	//Keeps adding states into queue until enough_states are reached
	generate_enough_states(queue);

	//Iterating over array underlying the queue
	//Dynamic scheduling provides better load balance because each configuration calculates different amount of time
	//Chunks of size 4 are optimal because each thread has 64 subproblems, which makes it fit nicely and it also lowers thread overhead
	#pragma omp parallel for schedule(dynamic, 4)
	for (uint64_t i = 0; i < queue.size(); ++i) {
		//Solve the remainder of configuration sequentially
		solve_seq(queue.at(i));
	}

	cout << "price: " << best_price << endl;
	cout << "configuration: ";
	print_configuration(best_configuration, n);
	
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
	
	//Makes sure there are enough problems to load balance
	enough_states = THRDS * 64;
	//Solve the problem
	solve_start();

	return 0;
}