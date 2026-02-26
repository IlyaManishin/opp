#pragma once

#include <stdio.h>
#include <time.h>

#define LOG_FILE "logger.log"

typedef enum
{
    LOG_INFO,
    LOG_WARN,
    LOG_ERROR
} LogLevel;

void log_init(int rank);
void log_message(LogLevel level, const char *msg);

#define LOG_TIME(stmt)                                                                                                     \
    do                                                                                                                     \
    {                                                                                                                      \
        clock_t start = clock();                                                                                           \
        stmt;                                                                                                              \
        clock_t end = clock();                                                                                             \
        char buf[100];                                                                                                     \
        snprintf(buf, sizeof(buf), "Execution time of `" #stmt "`: %.6f seconds", (double)(end - start) / CLOCKS_PER_SEC); \
        log_message(LOG_INFO, buf);                                                                                        \
    } while (0);
