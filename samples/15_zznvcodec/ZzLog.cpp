#include "ZzLog.h"

#include <stdio.h>
#include <stdarg.h> 

int QCAP_LOG_LEVEL = 0;
int64_t QCAP_LOG_START_TIME = -1;

ZzLog::ZzLog(int level, const char* tag) : level(level), tag(tag) {
}

void ZzLog::operator() (const char* fmt, ...) {
#if ! ZZLOG_HIDE
	if(QCAP_LOG_LEVEL > level) return;
	__ZZ_LOG_TIMESTAMP__();
	printf("\033%s: ", tag);
	__ZZ_LOG_POST__();
#endif
}