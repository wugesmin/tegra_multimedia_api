#ifndef __ZZ_LOG_H__
#define __ZZ_LOG_H__

#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>

namespace __zz_log__ {
	extern int QCAP_LOG_LEVEL;

	const static int QCAP_LOG_MAX_BUF_SIZE = 512;
	const static int QCAP_LOG_LEVEL_VERBOSE = 2;
	const static int QCAP_LOG_LEVEL_DEBUG = 3;
	const static int QCAP_LOG_LEVEL_INFO = 4;
	const static int QCAP_LOG_LEVEL_WARN = 5;
	const static int QCAP_LOG_LEVEL_ERROR = 6;
	const static int QCAP_LOG_LEVEL_NOTICE = 8;

	int FormatLog(char* buf, size_t buf_size, const char* prefix, const char* fmt, va_list& marker);
}

struct ZzLog {
	int level;
	const char* prefix;

	ZzLog(int level, const char* prefix);
	void operator() (const char* fmt, ...);
};

#if ZZLOG_HIDE
#define __ZZ_LOG_IMPL__(LEVEL, PREFIX, FMT)
#else // ZZLOG_HIDE
#define __ZZ_LOG_IMPL__(LEVEL, PREFIX, FMT) \
		if(__zz_log__::QCAP_LOG_LEVEL > LEVEL) return; \
		char _buf[__zz_log__::QCAP_LOG_MAX_BUF_SIZE]; \
		va_list marker; \
		va_start(marker, FMT); \
		int _buf_end = __zz_log__::FormatLog(_buf, sizeof(_buf), PREFIX, FMT, marker); \
		va_end(marker); \
		fwrite(_buf, _buf_end, 1, stdout)
#endif // ZZLOG_HIDE

#define ZZ_DECL_LOG_CLASS(TAG) \
	struct __LOGV__ : public ZzLog { __LOGV__() : \
		ZzLog(__zz_log__::QCAP_LOG_LEVEL_VERBOSE, "\033[0;37mVERBOSE[" TAG "]: ") {} }; \
	struct __LOGD__ : public ZzLog { __LOGD__() : \
		ZzLog(__zz_log__::QCAP_LOG_LEVEL_DEBUG, "\033[0;36mDEBUG[" TAG "]: ") {} }; \
	struct __LOGI__ : public ZzLog { __LOGI__() : \
		ZzLog(__zz_log__::QCAP_LOG_LEVEL_INFO, "\033[0mINFO[" TAG "]: ") {} }; \
	struct __LOGW__ : public ZzLog { __LOGW__() : \
		ZzLog(__zz_log__::QCAP_LOG_LEVEL_WARN, "\033[1;33mWARN[" TAG "]: ") {} }; \
	struct __LOGE__ : public ZzLog { __LOGE__() : \
		ZzLog(__zz_log__::QCAP_LOG_LEVEL_ERROR, "\033[1;31mERROR[" TAG "]: ") {} }; \
	struct __LOGN__ : public ZzLog { __LOGN__() : \
		ZzLog(__zz_log__::QCAP_LOG_LEVEL_NOTICE, "\033[1;32m[" TAG "]: ") {} }

#define ZZ_INIT_LOG(TAG) \
	ZZ_DECL_LOG_CLASS(TAG); \
	static __LOGV__ LOGV; \
	static __LOGD__ LOGD; \
	static __LOGI__ LOGI; \
	static __LOGW__ LOGW; \
	static __LOGE__ LOGE; \
	static __LOGN__ LOGN

#define TRACE_TAG() LOGI("\033[1;33m**** %s(%d)\033[0m", __FILE__, __LINE__)
#define TODO_TAG() LOGW("%s(%d): TODO", __FILE__, __LINE__)

#endif // __ZZ_LOG_H__