#ifndef __ZZ_LOG_H__
#define __ZZ_LOG_H__

#include <stdio.h>
#include <stdarg.h> 
#include <stdint.h>
#include <time.h>

extern int QCAP_LOG_LEVEL;
extern int64_t QCAP_LOG_START_TIME;

struct ZzLog {
	const char* tag;
	int level;

	ZzLog(int level, const char* tag);
	void operator() (const char* fmt, ...);
};

#define ZZ_DECL_LOG_CLASS(TAG) \
	struct __LOGV__ : public ZzLog { __LOGV__() : ZzLog(2, "[0;37mVERBOSE[" TAG "]") {} }; \
	struct __LOGD__ : public ZzLog { __LOGD__() : ZzLog(3, "[0;36mDEBUG[" TAG "]") {} }; \
	struct __LOGI__ : public ZzLog { __LOGI__() : ZzLog(4, "[0mINFO[" TAG "]") {} }; \
	struct __LOGW__ : public ZzLog { __LOGW__() : ZzLog(5, "[1;33mWARN[" TAG "]") {} }; \
	struct __LOGE__ : public ZzLog { __LOGE__() : ZzLog(6, "[1;31mERROR[" TAG "]") {} }; \
	struct __LOGN__ : public ZzLog { __LOGN__() : ZzLog(8, "[1;32m[" TAG "]") {} }

#define ZZ_DECL_LOG_VAR() \
	extern __LOGV__ _LOGV; \
	extern __LOGD__ _LOGD; \
	extern __LOGI__ _LOGI; \
	extern __LOGW__ _LOGW; \
	extern __LOGE__ _LOGE; \
	extern __LOGN__ _LOGN

#define ZZ_DEFINE_LOG_VAR() \
	__LOGV__ _LOGV; \
	__LOGD__ _LOGD; \
	__LOGI__ _LOGI; \
	__LOGW__ _LOGW; \
	__LOGE__ _LOGE; \
	__LOGN__ _LOGN

#define ZZ_INIT_LOG_VAR(TAG) \
	ZZ_DECL_LOG_CLASS(TAG); \
	ZZ_DEFINE_LOG_VAR()

#if ZZLOG_HIDE

#define ZZ_LOG_DEFINE(name, level, prefix, tag) \
	static void __log_print_ ## name (const char* fmt, ...) {}

#else

#define __ZZ_LOG_TIMESTAMP__() \
		struct timespec ts = { 0, 0 }; \
		clock_gettime(CLOCK_MONOTONIC, &ts); \
		int64_t curTimeMsec = ts.tv_sec*1000LL + ts.tv_nsec/1000000LL; \
		if (-1 == QCAP_LOG_START_TIME) \
			QCAP_LOG_START_TIME = curTimeMsec; \
		int64_t elapsedMsec = curTimeMsec - QCAP_LOG_START_TIME; \
		printf ("[%d:%02d:%02d:%03d] ", (int)(elapsedMsec/3600000LL), (int)((elapsedMsec/60000LL)%60LL), (int)((elapsedMsec/1000LL)%60LL), (int)(elapsedMsec%1000LL))

#define __ZZ_LOG_POST__() \
		va_list marker; \
		va_start(marker, fmt); \
		vprintf(fmt, marker); \
		va_end(marker); \
		printf("\n\033[0m")

#define ZZ_LOG_DEFINE(name, level, prefix, tag) \
	static void __log_print_ ## name (const char* fmt, ...) { \
		if(QCAP_LOG_LEVEL > level) return; \
		__ZZ_LOG_TIMESTAMP__(); \
		printf(prefix tag "]: "); \
		__ZZ_LOG_POST__(); \
	}
	
#endif

#define ZZ_INIT_LOG(TAG) \
	ZZ_LOG_DEFINE(verbose, 2,	"\033[0;37mVERBOSE[", TAG); \
	ZZ_LOG_DEFINE(debug, 3,		"\033[0;36mDEBUG[", TAG); \
	ZZ_LOG_DEFINE(info, 4,		"\033[0mINFO[", TAG); \
	ZZ_LOG_DEFINE(warn, 5,		"\033[1;33mWARN[", TAG); \
	ZZ_LOG_DEFINE(error, 6,		"\033[1;31mERROR[", TAG); \
	ZZ_LOG_DEFINE(notice, 8,	"\033[1;32m[", TAG);

#define LOGV(...) __log_print_verbose(__VA_ARGS__)
#define LOGD(...) __log_print_debug(__VA_ARGS__)
#define LOGI(...) __log_print_info(__VA_ARGS__)
#define LOGW(...) __log_print_warn(__VA_ARGS__)
#define LOGE(...) __log_print_error(__VA_ARGS__)
#define LOGN(...) __log_print_notice(__VA_ARGS__)

#define TRACE_TAG() LOGI("\033[1;33m**** %s(%d)\033[0m", __FILE__, __LINE__)

#endif // __ZZ_LOG_H__