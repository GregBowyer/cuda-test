#ifndef TIMER_H
#define TIMER_H 1

#include <cutil.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *name;
    unsigned int time;
} CUTimer;

/**
 * Starts a timer with the given name
 * @param *name - The name for the timer
 * @return CUTimer* - pointer to the newly created timer
 * @lifecycle - finish_timing destroys the timer, caller does not manage lifecycle
 */
extern CUTimer* start_timing(const char *name);

/**
 * Simple function to terminate an inprogress timer, printing out
 * the timer as well as performing the relevant cleanup
 * @param timer - the timer to manage
 * @return CUTBoolean as returned by cutStopTimer
 */
extern CUTBoolean finish_timing(CUTimer *timer);

#ifdef __cplusplus
}
#endif

#endif
