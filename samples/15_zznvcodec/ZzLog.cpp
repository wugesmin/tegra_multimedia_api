#include "ZzLog.h"
// #include "ZzUtils.h"

#include <time.h>

namespace __zz_log__ {
	int QCAP_LOG_LEVEL = 0;

	void _init_() {
		// QCAP_LOG_LEVEL = ZzUtils::GetEnvIntValue("QCAP_LOG_LEVEL", 4);
	}

	void _uninit_() {
	}

	int FormatLog(char* buf, size_t buf_size, const char* prefix, const char* fmt, va_list& marker) {
		struct timespec ts;
		clock_gettime(CLOCK_MONOTONIC, &ts);
		int64_t curTimeMsec = ts.tv_sec*1000LL + ts.tv_nsec/1000000LL;
		int buf_end = snprintf(buf, buf_size, "[%d:%02d:%02d:%03d] %s",
			(int)(curTimeMsec/3600000LL), (int)((curTimeMsec/60000LL)%60LL),
			(int)((curTimeMsec/1000LL)%60LL), (int)(curTimeMsec%1000LL), prefix);
		buf_end += vsnprintf(buf + buf_end, buf_size - buf_end, fmt, marker);
		buf_end += snprintf(buf + buf_end, buf_size - buf_end, "\n\033[0m");

		return buf_end;
	}
}

ZzLog::ZzLog(int level, const char* prefix) :
	level(level), prefix(prefix) {
}

void ZzLog::operator() (const char* fmt, ...) {
	__ZZ_LOG_IMPL__(level, prefix, fmt);
}