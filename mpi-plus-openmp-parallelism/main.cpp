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
#include <thread>
#include <chrono>
#include <omp.h>
#include <fstream>
#include <mpi.h>
#include <algorithm>

using namespace std;

//Declarations
//Maximum N is currently 64, the program can be simly modified to use N>64 by changing MAX_N as well as the configuration type (for example to bitset<N>...)
typedef uint64_t configuration_type;
#define MAX_N 64
#define WORK_TERMINATE -999

//ReadWrite Atomics (need to be written to by a single thread at a time)
//Best configuration globally
configuration_type best_configuration = 0;
//Best price, starts at max
double best_price = numeric_limits<double>::max();
//Counter for amount of recursive function calls
atomic<uint64_t> counter(0);
//Maximum states for parallel for, value is adjusted dynamically in main()
uint64_t enough_states_mp = 500;
//Amount of threads to run MP on, configured in main
uint64_t use_threads = 1;

//Read Atomics (implicit, require read atomicity)
//N, K, A parameters from user input
int64_t n, k, a;
//Keeping graph on stack (as matrix)
double graph[MAX_N][MAX_N];
//Maximum states for MPI, value is adjusted dynamically in main()
uint64_t enough_states_mpi = 500;


//OpenMPI rank
int cpu_rank = 0;
//OpenMPI amount of processes
int num_procs = 0;
//OpenMPI amount of work sent to slaves
int work_sent = 0;

//Custom OpenMPI struct for communication
int blocklengths[3] = {1, 1, 1};
MPI_Datatype types[3] = { MPI_UINT64_T, MPI_INT64_T, MPI_DOUBLE };
MPI_Datatype problem_type;
MPI_Aint offsets[3];

//Struct has different meaning when send by master and by slave
struct single_problem {
	//master: configuration to solve, slave: best configuration found
	uint64_t configuration;
	//master: index at which solve should start, slave: rank of slave
	int64_t index;
	//master: best price found (for B&B), slave: best price of given configuration
	double current_best_price;
};


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
//From data paralelism
void solve_seq(configuration_type configuration, int64_t index) noexcept {
	double current_price = 0;
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
void generate_enough_states(IterativeQueue & queue, uint64_t required_states) noexcept {
	//Create enough states, also check for bad settings that could create problems
	while (queue.size() <= required_states) {
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

//Start solving by slave (MP)
void solve_start(configuration_type configuration, int64_t index) noexcept {
	//Create queue that uses array underneath
	IterativeQueue queue(enough_states_mp);
	//Push configuration that doesn't exist but works for generating others
	queue.push(move(make_pair(configuration, index)));
	//Keeps adding states into queue until enough_states are reached
	generate_enough_states(queue, enough_states_mp);

	//Iterating over array underlying the queue
	//Dynamic scheduling provides better load balance because each configuration calculates different amount of time
	//Chunks of size 4 are optimal because each thread has 64 subproblems, which makes it fit nicely and it also lowers thread overhead
	#pragma omp parallel for schedule(dynamic, 4) num_threads(use_threads)
	for (uint64_t i = 0; i < queue.size(); ++i) {
		//Solve the remainder of configuration sequentially
		solve_seq(queue.at(i).first, queue.at(i).second);
	}
	#ifdef DEBUG
		cout << "function calls: " << counter << endl;
	#endif
}

//Start solving by master-slave (openMPI)
void solve_start_mpi() noexcept {
	//Create queue that uses array underneath
	IterativeQueue queue(enough_states_mpi);
	//Push configuration that doesn't exist but works for generating others
	queue.push(move(make_pair(0, -1)));
	//Keeps adding states into queue until enough_states are reached
	generate_enough_states(queue, enough_states_mpi);

	//Send one problem to each slave if there is enough problems
	int i = 1;
	for (; i < num_procs; ++i) {
		if (queue.size() <= 0) {
			break;
		}
		//Get problem from queue
		pair<uint64_t, int64_t> subproblem = queue.pop();
		//Put it into struct that is being sent to slave along with current best_price
		single_problem p = {subproblem.first, subproblem.second, best_price};
		//Send one to every slave
		MPI_Send(&p, 1, problem_type, i, 0, MPI_COMM_WORLD);
		//Keep track of how many are sent out
		++work_sent;
	}

	//In case there are not enough problems (shouldn't happen if parameters are set correctly)
	if (i < num_procs && queue.size() <= 0) {
		cout << "Need more problems next time." << endl;

		for (int j = 0; j < i; ++j) {
			single_problem p;
			MPI_Status status;
			//Get all answers
			MPI_Recv(&p, 1, problem_type, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			//Make sure we got all of them
			--work_sent;
			//Update price if necessary
			if (p.current_best_price < best_price) {
				best_price = p.current_best_price;
				best_configuration = p.configuration;
			}
		}
	//There are still more subproblems to solve
	} else {
		//While slaves are working
		while(work_sent > 0) {
			single_problem p;
			MPI_Status status;
			//Receive solution from any slave
			MPI_Recv(&p, 1, problem_type, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			//Keep track of amount of work slaves have, if this reaches zero and there is no more work to send, all work has been done
			--work_sent;
			//Update price if better configuration found
			if (p.current_best_price < best_price) {
				best_price = p.current_best_price;
				best_configuration = p.configuration;
			}
			//If there is still work to be done, send him another problem
			if (queue.size() > 0) {
				pair<uint64_t, int64_t> subproblem = queue.pop();
				single_problem s = {subproblem.first, subproblem.second, best_price};
				MPI_Send(&s, 1, problem_type, p.index, 0, MPI_COMM_WORLD);
				++work_sent;
			}
		}
	}

	//After all work is done, send special value that terminates all slaves
	for (int i = 1; i < num_procs; ++i) {
		single_problem p = {0, WORK_TERMINATE, 0};
		MPI_Send(&p, 1, problem_type, i, 0, MPI_COMM_WORLD);
	}

	//Extra safety check, should not be needed
	if (work_sent == 0) {
		cout << "All branches calculated." << endl;
	}
	//Print solution
	cout << "price: " << best_price << endl;
	cout << "configuration: ";
	print_configuration(best_configuration, n);
	
	#ifdef DEBUG
		cout << "function calls: " << counter << endl;
	#endif
}

//HOW TO RUN
//mpirun -np num_of_mpi_processes --mca mpi_yield_when_idle 1 a.out "problem_file" "num_of_mp_threads"
//mpi_yield allows to run one extra process on the same core as master since master is asleep when blocking (default is active waiting)
int main(int argc, char * argv[]) {
	//MPI init
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD , &cpu_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	//Init of struct we will use to send data between master and slaves
	offsets[0] = offsetof(single_problem, configuration);
    offsets[1] = offsetof(single_problem, index);
    offsets[2] = offsetof(single_problem, current_best_price);
    MPI_Type_create_struct(3, blocklengths, offsets, types, &problem_type);
    MPI_Type_commit(&problem_type);

	//We need at least one argument (file with problem)
	if (argc < 2) {
		cout << "Invalid args." << endl;
		return 2;
	}
	//Try to open it
	ifstream input;
	input.open(argv[1]);
	if (!input) {
		cout << "Unable to read file" << endl;
		return 3;
	}

	//If there is another argument, it serves as number of threads used for MP, otherwise, use everything available
	if (argc != 3) {
		use_threads = omp_get_max_threads();
	} else {
		use_threads = stoi(argv[2]);
	}

	string tmp_line = "";
	//Load first line of file into memory
	{
		getline(input, tmp_line);
		stringstream parser(tmp_line);
		parser >> n >> k >> a;		
	}

	//Load graph into matrix (fast loading into memory, then parsing)
	int i = 0, j = 0;
	double price = 0.0;
	while (getline(input, tmp_line)) {
		stringstream parser(tmp_line);
		parser >> i >> j >> price;
		graph[i][j] = price;
		graph[j][i] = price;
	}


	//Makes sure there are enough problems to load balance, 64 was tested to work well with data parallelism
	enough_states_mp = use_threads * 64;

	//MASTER
	if (cpu_rank == 0) {
		//Give some info about what resources we are using
		cout << "mpi processes used for computing: " << num_procs - 1 << " (+ 1 sleeping master)" << endl;
		cout << "mp threads per process used for computing: " << use_threads << endl;
		
		//Makes sure there are enough problems to load balance, for master-slave we want to avoid communication
		//but still have enough problems for B&B price cut through and load balance
		//Every time new problem is sent to slave, best price is also updated (better price B&B efficiency)
		enough_states_mpi = num_procs * 64;
		//Solve the problem by master-slave
		solve_start_mpi();
	//SLAVE
	} else {
		while (true) {
			//Receive problem from master
			single_problem p;
        	MPI_Status status;
			MPI_Recv(&p, 1, problem_type, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
			//If there is no more work to be done, terminate
			if (p.index == WORK_TERMINATE) {
				break;
			}
			//Update best_price for B&B
			best_price = p.current_best_price;
			//Solve with MP
			solve_start(p.configuration, p.index);
			//Get results ready
			p.configuration = best_configuration;
			p.current_best_price = best_price;
			//Sending cpu rank in index so the master knows who to send more work to
			p.index = cpu_rank;
			//Send results back
			MPI_Send(&p, 1, problem_type, 0, 0, MPI_COMM_WORLD);
		}
	}

	MPI_Finalize();
	return 0;
}