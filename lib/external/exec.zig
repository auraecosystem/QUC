ZEXTERN const char   * ZEXPORT zError(int);
ZEXTERN int            ZEXPORT inflateSyncPoint(z_streamp z);
ZEXTERN const z_crc_t FAR * ZEXPORT get_crc_table(void);
ZEXTERN int            ZEXPORT inflateUndermine(z_streamp, int);
ZEXTERN int            ZEXPORT inflateValidate(z_streamp, int);
ZEXTERN unsigned long  ZEXPORT inflateCodesUsed(z_streamp);
ZEXTERN int            ZEXPORT inflateResetKeep(z_streamp);
ZEXTERN int            ZEXPORT deflateResetKeep(z_streamp);
#if defined(_WIN32) && !defined(Z_SOLO)
ZEXTERN gzFile         ZEXPORT gzopen_w(const wchar_t *path,
                                        const char *mode);
#endif
#if defined(STDC) || defined(Z_HAVE_STDARG_H)
#  ifndef Z_SOLO
ZEXTERN int            ZEXPORTVA gzvprintf(gzFile file,
                                           const char *format,
                                           va_list va);
#  endif
#endif
