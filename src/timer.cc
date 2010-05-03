#include "timer.h"

CUTimer* start_timing(const char *name) {
    CUTimer *timer = (CUTimer *) malloc(sizeof(CUTimer));
    timer->time = 0;
    timer->name = name;
    cutCreateTimer(&timer->time);
    cutStartTimer(timer->time);
    return timer;
}

CUTBoolean finish_timing(CUTimer *timer) {
    cutStopTimer(timer->time);
    printf("%s took: %0.3f ms\n", timer->name, cutGetTimerValue(timer->time));
    CUTBoolean to_return = cutDeleteTimer(timer->time);
    free(timer);
    return to_return;
}
