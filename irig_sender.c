#define _POSIX_C_SOURCE 199309L
#define _GNU_SOURCE
#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <sched.h>
#include <stdint.h>

#define BCM2708_PERI_BASE_RPI1  0x20000000
#define BCM2708_PERI_BASE_RPI2  0x3F000000
#define BCM2708_PERI_BASE_RPI4  0xFE000000
#define GPIO_BASE_OFFSET        0x200000

#define BLOCK_SIZE (4*1024)

#define GPFSEL0   0x00
#define GPFSEL1   0x04
#define GPFSEL2   0x08
#define GPSET0    0x1C
#define GPCLR0    0x28
#define GPLEV0    0x34

static volatile unsigned *gpio_map = NULL;
static int gpio_mem_fd = -1;

typedef enum {
    IRIG_ZERO = 0,
    IRIG_ONE = 1,
    IRIG_P = 2
} irig_bit_t;

typedef struct {
    double *data;
    size_t length;
    size_t capacity;
} double_array_t;

#define SENDING_BIT_LENGTH 1.0
// Offset to account for pin toggle latency (tuned via oscilloscope)
#define OFFSET_NS 20000
// Sleep until this much time before target, then busy wait for precision
#define BUSY_WAIT_BUFFER_NS 1000000L
// Sleep interval during busy wait (0 = pure busy wait for maximum precision)
#define BUSY_WAIT_SLEEP_NS 0L

static const uint64_t NS_PER_SEC = 1000000000ULL;
static uint64_t bit_length_ns;

static const int SECONDS_WEIGHTS[] = {1, 2, 4, 8, 10, 20, 40};
static const int MINUTES_WEIGHTS[] = {1, 2, 4, 8, 10, 20, 40};
static const int HOURS_WEIGHTS[] = {1, 2, 4, 8, 10, 20};
static const int DAY_OF_YEAR_WEIGHTS[] = {1, 2, 4, 8, 10, 20, 40, 80, 100, 200};
static const int DECISECONDS_WEIGHTS[] = {1, 2, 4, 8};
static const int YEARS_WEIGHTS[] = {1, 2, 4, 8, 10, 20, 40, 80};

typedef struct {
    int sending_gpio_pin;
    int inverted_gpio_pin;
    pthread_t sender_thread;
    bool running;
    double_array_t encoded_times;
    double_array_t sending_starts;
    char timestamp_filename[256];

    irig_bit_t current_frame[60];
    irig_bit_t next_frame[60];
    double pulse_lengths[60];
    uint64_t bit_start_times[60];
    struct timespec frame_start_time;
    volatile unsigned *gpio_set_reg;
    volatile unsigned *gpio_clr_reg;
    uint32_t gpio_mask;
    uint32_t inverted_gpio_mask;
} irig_h_sender_t;

void init_timing_constants(void);
uint64_t timespec_to_ns(const struct timespec *ts);
void ultra_wait_until_ns(uint64_t target_ns);

volatile sig_atomic_t running = 1;
static int debug_mode = 0;

void signal_handler(int sig) {
    printf("Received signal %d, shutting down gracefully...\n", sig);
    running = 0;
}

int detect_pi_model() {
    FILE *fp = fopen("/proc/device-tree/model", "r");
    if (!fp) return 2;
    
    char model[256];
    if (fgets(model, sizeof(model), fp)) {
        fclose(fp);
        if (strstr(model, "Raspberry Pi 4") || strstr(model, "Raspberry Pi 400")) {
            return 4;
        } else if (strstr(model, "Raspberry Pi 2") || strstr(model, "Raspberry Pi 3")) {
            return 2;
        } else {
            return 1;
        }
    }
    fclose(fp);
    return 2;
}

int gpio_init() {
    int pi_model = detect_pi_model();
    unsigned gpio_base;
    
    switch (pi_model) {
        case 1:
            gpio_base = BCM2708_PERI_BASE_RPI1 + GPIO_BASE_OFFSET;
            break;
        case 4:
            gpio_base = BCM2708_PERI_BASE_RPI4 + GPIO_BASE_OFFSET;
            break;
        default:
            gpio_base = BCM2708_PERI_BASE_RPI2 + GPIO_BASE_OFFSET;
            break;
    }
    
    printf("Detected Raspberry Pi model %d, using GPIO base 0x%08X\n", pi_model, gpio_base);

    gpio_mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (gpio_mem_fd < 0) {
        printf("Error: Cannot open /dev/mem. Are you running as root?\n");
        return -1;
    }
    
    gpio_map = (volatile unsigned *)mmap(
        NULL,
        BLOCK_SIZE,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        gpio_mem_fd,
        gpio_base
    );
    
    if (gpio_map == MAP_FAILED) {
        printf("Error: mmap failed: %s\n", strerror(errno));
        close(gpio_mem_fd);
        return -1;
    }
    
    printf("GPIO memory mapped successfully\n");
    return 0;
}

void gpio_cleanup() {
    if (gpio_map != NULL && gpio_map != MAP_FAILED) {
        munmap((void*)gpio_map, BLOCK_SIZE);
        gpio_map = NULL;
    }
    if (gpio_mem_fd >= 0) {
        close(gpio_mem_fd);
        gpio_mem_fd = -1;
    }
}

void gpio_set_output(int pin) {
    if (gpio_map == NULL) return;
    
    int reg = pin / 10;
    int shift = (pin % 10) * 3;

    *(gpio_map + reg) &= ~(7 << shift);
    *(gpio_map + reg) |= (1 << shift);
    
    printf("GPIO %d set as output\n", pin);
}

void gpio_write(int pin, int value) {
    if (gpio_map == NULL) return;

    if (value) {
        *(gpio_map + GPSET0/4) = 1 << pin;
    } else {
        *(gpio_map + GPCLR0/4) = 1 << pin;
    }
}

// Cache GPIO registers for fast access
void init_gpio_cache(irig_h_sender_t *sender) {
    if (gpio_map == NULL) return;

    sender->gpio_set_reg = gpio_map + GPSET0/4;
    sender->gpio_clr_reg = gpio_map + GPCLR0/4;
    sender->gpio_mask = 1 << sender->sending_gpio_pin;
    sender->inverted_gpio_mask = 1 << sender->inverted_gpio_pin;
}

double_array_t* create_double_array(size_t initial_capacity) {
    double_array_t *arr = malloc(sizeof(double_array_t));
    arr->data = malloc(sizeof(double) * initial_capacity);
    arr->length = 0;
    arr->capacity = initial_capacity;
    return arr;
}

void append_double(double_array_t *arr, double value) {
    if (arr->length >= arr->capacity) {
        arr->capacity *= 2;
        arr->data = realloc(arr->data, sizeof(double) * arr->capacity);
    }
    arr->data[arr->length++] = value;
}

void free_double_array(double_array_t *arr) {
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

void bcd_encode(int value, const int *weights, int weight_count, int *result) {
    memset(result, 0, weight_count * sizeof(int));
    for (int i = weight_count - 1; i >= 0; i--) {
        if (weights[i] <= value) {
            result[i] = 1;
            value -= weights[i];
        }
    }
}

void generate_irig_h_frame(irig_h_sender_t *sender, struct tm *time_info, irig_bit_t *frame) {
    append_double(&sender->encoded_times, (double)mktime(time_info));

    int seconds_bcd[7], minutes_bcd[7], hours_bcd[6];
    int day_of_year_bcd[10], deciseconds_bcd[4], year_bcd[8];
    
    bcd_encode(time_info->tm_sec + 1, SECONDS_WEIGHTS, 7, seconds_bcd);
    bcd_encode(time_info->tm_min, MINUTES_WEIGHTS, 7, minutes_bcd);
    bcd_encode(time_info->tm_hour, HOURS_WEIGHTS, 6, hours_bcd);
    bcd_encode(time_info->tm_yday + 1, DAY_OF_YEAR_WEIGHTS, 10, day_of_year_bcd);
    bcd_encode(0, DECISECONDS_WEIGHTS, 4, deciseconds_bcd);
    bcd_encode((time_info->tm_year + 1900) % 100, YEARS_WEIGHTS, 8, year_bcd);

    int pos = 0;
    frame[pos++] = IRIG_P;

    for (int i = 0; i < 4; i++) frame[pos++] = seconds_bcd[i] ? IRIG_ONE : IRIG_ZERO;
    frame[pos++] = IRIG_ZERO;
    for (int i = 4; i < 7; i++) frame[pos++] = seconds_bcd[i] ? IRIG_ONE : IRIG_ZERO;
    frame[pos++] = IRIG_P;

    for (int i = 0; i < 4; i++) frame[pos++] = minutes_bcd[i] ? IRIG_ONE : IRIG_ZERO;
    frame[pos++] = IRIG_ZERO;
    for (int i = 4; i < 7; i++) frame[pos++] = minutes_bcd[i] ? IRIG_ONE : IRIG_ZERO;
    frame[pos++] = IRIG_ZERO;
    frame[pos++] = IRIG_P;

    for (int i = 0; i < 4; i++) frame[pos++] = hours_bcd[i] ? IRIG_ONE : IRIG_ZERO;
    frame[pos++] = IRIG_ZERO;
    for (int i = 4; i < 6; i++) frame[pos++] = hours_bcd[i] ? IRIG_ONE : IRIG_ZERO;
    frame[pos++] = IRIG_ZERO; frame[pos++] = IRIG_ZERO;
    frame[pos++] = IRIG_P;

    for (int i = 0; i < 4; i++) frame[pos++] = day_of_year_bcd[i] ? IRIG_ONE : IRIG_ZERO;
    frame[pos++] = IRIG_ZERO;
    for (int i = 4; i < 8; i++) frame[pos++] = day_of_year_bcd[i] ? IRIG_ONE : IRIG_ZERO;
    frame[pos++] = IRIG_P;
    for (int i = 8; i < 10; i++) frame[pos++] = day_of_year_bcd[i] ? IRIG_ONE : IRIG_ZERO;

    frame[pos++] = IRIG_ZERO; frame[pos++] = IRIG_ZERO; frame[pos++] = IRIG_ZERO;
    for (int i = 0; i < 4; i++) frame[pos++] = deciseconds_bcd[i] ? IRIG_ONE : IRIG_ZERO;
    frame[pos++] = IRIG_P;

    for (int i = 0; i < 4; i++) frame[pos++] = year_bcd[i] ? IRIG_ONE : IRIG_ZERO;
    frame[pos++] = IRIG_ZERO;
    for (int i = 4; i < 8; i++) frame[pos++] = year_bcd[i] ? IRIG_ONE : IRIG_ZERO;
    frame[pos++] = IRIG_P;
}

void init_timing_constants(void) {
    bit_length_ns = (uint64_t)(SENDING_BIT_LENGTH * NS_PER_SEC);
}

uint64_t timespec_to_ns(const struct timespec *ts) {
    return (uint64_t)ts->tv_sec * NS_PER_SEC + (uint64_t)ts->tv_nsec;
}

void ultra_wait_until_ns(uint64_t target_ns) {
    struct timespec current_time;
    uint64_t current_ns;
    int64_t remaining_ns;
    struct timespec sleep_time;

    while (running) {
        clock_gettime(CLOCK_REALTIME, &current_time);
        current_ns = timespec_to_ns(&current_time);
        remaining_ns = (int64_t)(target_ns - current_ns);

        if (remaining_ns <= 0) break;

        if (remaining_ns > BUSY_WAIT_BUFFER_NS) {
            uint64_t sleep_duration = remaining_ns - BUSY_WAIT_BUFFER_NS;

            sleep_time.tv_sec = sleep_duration / NS_PER_SEC;
            sleep_time.tv_nsec = sleep_duration % NS_PER_SEC;
            nanosleep(&sleep_time, NULL);
        } else {
            break;
        }
    }

    #if BUSY_WAIT_SLEEP_NS > 0
        sleep_time.tv_sec = 0;
        sleep_time.tv_nsec = BUSY_WAIT_SLEEP_NS;
        do {
            clock_gettime(CLOCK_REALTIME, &current_time);
            current_ns = (uint64_t)current_time.tv_sec * NS_PER_SEC + (uint64_t)current_time.tv_nsec;
            if (current_ns < target_ns) {
                nanosleep(&sleep_time, NULL);
            }
        } while (current_ns < target_ns && running);
    #else
        do {
            clock_gettime(CLOCK_REALTIME, &current_time);
            current_ns = (uint64_t)current_time.tv_sec * NS_PER_SEC + (uint64_t)current_time.tv_nsec;
        } while (current_ns < target_ns && running);
    #endif
}

double calculate_pulse_length(irig_bit_t bit) {
    switch (bit) {
        case IRIG_P: return 0.8 * SENDING_BIT_LENGTH;
        case IRIG_ONE: return 0.5 * SENDING_BIT_LENGTH;
        case IRIG_ZERO:
        default: return 0.2 * SENDING_BIT_LENGTH;
    }
}

void ultra_fast_pulse(irig_h_sender_t *sender, uint64_t pulse_duration_ns) {
    struct timespec start_time, current_time, sleep_time;
    uint64_t start_ns, target_ns, current_ns;
    int64_t remaining_ns;

    clock_gettime(CLOCK_REALTIME, &start_time);
    start_ns = timespec_to_ns(&start_time);
    target_ns = start_ns + pulse_duration_ns;

    *(sender->gpio_set_reg) = sender->gpio_mask;
    *(sender->gpio_clr_reg) = sender->inverted_gpio_mask;

    while (running) {
        clock_gettime(CLOCK_REALTIME, &current_time);
        current_ns = timespec_to_ns(&current_time);
        remaining_ns = (int64_t)(target_ns - current_ns);

        if (remaining_ns <= 0) break;

        if (remaining_ns > BUSY_WAIT_BUFFER_NS) {
            uint64_t sleep_duration = remaining_ns - BUSY_WAIT_BUFFER_NS;
            sleep_time.tv_sec = sleep_duration / NS_PER_SEC;
            sleep_time.tv_nsec = sleep_duration % NS_PER_SEC;
            nanosleep(&sleep_time, NULL);
        } else {
            break;
        }
    }

    #if BUSY_WAIT_SLEEP_NS > 0
        struct timespec poll_sleep;
        poll_sleep.tv_sec = 0;
        poll_sleep.tv_nsec = BUSY_WAIT_SLEEP_NS;

        do {
            clock_gettime(CLOCK_REALTIME, &current_time);
            current_ns = (uint64_t)current_time.tv_sec * NS_PER_SEC + (uint64_t)current_time.tv_nsec;
            if (current_ns < target_ns) {
                nanosleep(&poll_sleep, NULL);
            }
        } while (current_ns < target_ns && running);
    #else
        do {
            clock_gettime(CLOCK_REALTIME, &current_time);
            current_ns = (uint64_t)current_time.tv_sec * NS_PER_SEC + (uint64_t)current_time.tv_nsec;
        } while (current_ns < target_ns && running);
    #endif

    *(sender->gpio_clr_reg) = sender->gpio_mask;
    *(sender->gpio_set_reg) = sender->inverted_gpio_mask;
}

// Pre-calculate timing for the next frame
void precalculate_next_frame(irig_h_sender_t *sender, time_t target_second) {
    struct tm *time_info = localtime(&target_second);
    generate_irig_h_frame(sender, time_info, sender->next_frame);

    uint64_t frame_start_ns = (uint64_t)target_second * NS_PER_SEC;
    for (int i = 0; i < 60; i++) {
        sender->pulse_lengths[i] = calculate_pulse_length(sender->next_frame[i]);
        sender->bit_start_times[i] = frame_start_ns + (i * NS_PER_SEC) - OFFSET_NS;
    }
}

void* continuous_irig_sending(void *arg) {
    irig_h_sender_t *sender = (irig_h_sender_t*)arg;
    struct timespec current_time;
    time_t current_second, next_second;
    bool frame_ready = false;

    static int constants_initialized = 0;
    if (!constants_initialized) {
        init_timing_constants();
        constants_initialized = 1;
    }

    struct sched_param param;
    param.sched_priority = 80;
    if (sched_setscheduler(0, SCHED_FIFO, &param) < 0) {
        printf("Warning: Could not set SCHED_FIFO: %s (continuing with normal scheduling)\n", strerror(errno));
    } else {
        printf("Set SCHED_FIFO with priority 80\n");
    }

    printf("IRIG-H continuous transmission thread started\n");

    clock_gettime(CLOCK_REALTIME, &current_time);
    current_second = current_time.tv_sec;
    next_second = current_second + 1;
    precalculate_next_frame(sender, next_second);
    
    while (sender->running && running) {
        clock_gettime(CLOCK_REALTIME, &current_time);

        long ns_into_second = current_time.tv_nsec;
        if (ns_into_second >= 800000000L && !frame_ready) {
            time_t upcoming_second = current_time.tv_sec + 1;
            precalculate_next_frame(sender, upcoming_second);
            frame_ready = true;
        }

        if (current_time.tv_sec > current_second) {
            current_second = current_time.tv_sec;

            memcpy(sender->current_frame, sender->next_frame, sizeof(sender->current_frame));

            double start_time_double = (double)current_second + (double)current_time.tv_nsec * 1e-9;
            append_double(&sender->sending_starts, start_time_double);

            for (int i = 0; i < 60 && sender->running && running; i++) {
                uint64_t bit_start_target = sender->bit_start_times[i];
                ultra_wait_until_ns(bit_start_target);

                if (debug_mode) {
                    struct timespec current;
                    clock_gettime(CLOCK_REALTIME, &current);
                    printf("Bit %d sent at system time: %ld.%09ld\n", i, current.tv_sec, current.tv_nsec);
                }

                uint64_t pulse_ns = (uint64_t)(sender->pulse_lengths[i] * NS_PER_SEC);
                ultra_fast_pulse(sender, pulse_ns);
            }

            frame_ready = false;
        }

        usleep(10000);
    }

    printf("IRIG-H transmission thread stopping\n");
    return NULL;
}

irig_h_sender_t* create_irig_h_sender(int gpio_pin, int inverted_gpio_pin) {
    irig_h_sender_t *sender = malloc(sizeof(irig_h_sender_t));
    
    sender->sending_gpio_pin = gpio_pin;
    sender->inverted_gpio_pin = inverted_gpio_pin;
    sender->running = false;
    
    sender->encoded_times.data = malloc(sizeof(double) * 100);
    sender->encoded_times.length = 0;
    sender->encoded_times.capacity = 100;
    
    sender->sending_starts.data = malloc(sizeof(double) * 100);
    sender->sending_starts.length = 0;
    sender->sending_starts.capacity = 100;
    
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    strftime(sender->timestamp_filename, sizeof(sender->timestamp_filename),
             "irig_output_timestamps_%Y-%m-%d_%H-%M-%S.csv", tm_info);

    if (gpio_init() < 0) {
        printf("Failed to initialize GPIO hardware access\n");
        free(sender->encoded_times.data);
        free(sender->sending_starts.data);
        free(sender);
        return NULL;
    }

    gpio_set_output(sender->sending_gpio_pin);
    gpio_set_output(sender->inverted_gpio_pin);

    init_gpio_cache(sender);

    gpio_write(sender->sending_gpio_pin, 0);
    gpio_write(sender->inverted_gpio_pin, 1);

    return sender;
}

void start_irig_sender(irig_h_sender_t *sender) {
    sender->running = true;
    pthread_create(&sender->sender_thread, NULL, continuous_irig_sending, sender);
}

void write_timestamps_to_file(irig_h_sender_t *sender) {
    FILE *file = fopen(sender->timestamp_filename, "w");
    if (!file) {
        printf("Could not open file for writing: %s\n", sender->timestamp_filename);
        return;
    }
    
    fprintf(file, "Encoded times,Sending starts\n");
    size_t min_length = (sender->encoded_times.length < sender->sending_starts.length) 
                       ? sender->encoded_times.length : sender->sending_starts.length;
    
    for (size_t i = 0; i < min_length; i++) {
        fprintf(file, "%f,%f\n", sender->encoded_times.data[i], sender->sending_starts.data[i]);
    }
    
    fclose(file);
    printf("Timestamps written to %s\n", sender->timestamp_filename);
}

void finish_irig_sender(irig_h_sender_t *sender) {
    sender->running = false;
    pthread_join(sender->sender_thread, NULL);

    gpio_write(sender->sending_gpio_pin, 0);
    gpio_write(sender->inverted_gpio_pin, 1);

    gpio_cleanup();

    free(sender->encoded_times.data);
    free(sender->sending_starts.data);
    free(sender);
}

int main(int argc, char *argv[]) {
    if (argc > 1 && strcmp(argv[1], "debug") == 0) {
        debug_mode = 1;
        printf("Debug mode enabled\n");
    }

    signal(SIGTERM, signal_handler);
    signal(SIGINT, signal_handler);

    printf("IRIG-H Timecode Sender starting on GPIO 17 (with inverted output on GPIO 27)...\n");
    printf("Using direct hardware register access\n");

    irig_h_sender_t *sender = create_irig_h_sender(17, 27);
    if (!sender) {
        printf("Failed to initialize IRIG-H sender\n");
        return 1;
    }

    printf("IRIG-H sender initialized successfully\n");
    start_irig_sender(sender);
    printf("IRIG-H transmission started on GPIO 17 (inverted on GPIO 27)\n");

    while (running) {
        sleep(1);
    }

    printf("Stopping IRIG-H sender...\n");
    finish_irig_sender(sender);
    printf("IRIG-H sender stopped\n");

    return 0;
}