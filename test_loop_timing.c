// Test program to measure busy wait loop iterations
// Compile: gcc -o test_loop_timing test_loop_timing.c
// Run: sudo ./test_loop_timing

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <time.h>
#include <stdint.h>

#define NS_PER_SEC 1000000000ULL

int main() {
    struct timespec current_time, start_time;
    uint64_t target_ns, current_ns, start_ns;
    int nruns = 0;

    // Test different wait durations
    uint64_t test_durations[] = {1000, 5000, 10000, 20000, 50000}; // nanoseconds

    for (int test = 0; test < 5; test++) {
        clock_gettime(CLOCK_REALTIME, &start_time);
        start_ns = (uint64_t)start_time.tv_sec * NS_PER_SEC + (uint64_t)start_time.tv_nsec;
        target_ns = start_ns + test_durations[test];

        nruns = 0;
        do {
            clock_gettime(CLOCK_REALTIME, &current_time);
            current_ns = (uint64_t)current_time.tv_sec * NS_PER_SEC + (uint64_t)current_time.tv_nsec;
            nruns++;
        } while (current_ns < target_ns);

        uint64_t actual_duration = current_ns - start_ns;

        printf("Target wait: %lu ns, Actual: %lu ns, Loop iterations: %d, Avg per iteration: %.1f ns\n",
               test_durations[test], actual_duration, nruns,
               (double)actual_duration / nruns);
    }

    printf("\n--- Testing with function call (original implementation) ---\n");

    uint64_t timespec_to_ns(const struct timespec *ts) {
        return (uint64_t)ts->tv_sec * NS_PER_SEC + (uint64_t)ts->tv_nsec;
    }

    for (int test = 0; test < 5; test++) {
        clock_gettime(CLOCK_REALTIME, &start_time);
        start_ns = timespec_to_ns(&start_time);
        target_ns = start_ns + test_durations[test];

        nruns = 0;
        do {
            clock_gettime(CLOCK_REALTIME, &current_time);
            nruns++;
        } while (timespec_to_ns(&current_time) < target_ns);

        clock_gettime(CLOCK_REALTIME, &current_time);
        uint64_t actual_duration = timespec_to_ns(&current_time) - start_ns;

        printf("Target wait: %lu ns, Actual: %lu ns, Loop iterations: %d, Avg per iteration: %.1f ns\n",
               test_durations[test], actual_duration, nruns,
               (double)actual_duration / nruns);
    }

    return 0;
}
