#include "logger.h"

#include <stdio.h>
#include <time.h>

static int _rank = 0;
static const char *level_names[] = {"INFO", "WARN", "ERROR"};

void log_init(int rank)
{
    _rank = rank;
}

void log_message(LogLevel level, const char *msg)
{
    FILE *f = fopen(LOG_FILE, "a");
    if (!f)
        return;

    time_t t = time(NULL);
    struct tm *tm_info = localtime(&t);
    char time_buf[20];
    strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", tm_info);

    fprintf(f, "%d: [%s] %s: %s\n", _rank, time_buf, level_names[level], msg);
    fclose(f);
}