import sys
from cffi import FFI

# Initialize CFFI interface
ffi = FFI()

# 1. Register C Declarations & Structs
ffi.cdef("""
    typedef unsigned char Bytef;
    typedef unsigned int uInt;
    typedef unsigned long uLong;
    typedef void* voidpf;

    typedef voidpf (*alloc_func)(voidpf opaque, uInt items, uInt size);
    typedef void   (*free_func)(voidpf opaque, voidpf address);

    struct internal_state;

    typedef struct z_stream_s {
        const Bytef *next_in;
        uInt     avail_in;
        uLong    total_in;

        Bytef    *next_out;
        uInt     avail_out;
        uLong    total_out;

        const char *msg;
        struct internal_state *state;

        alloc_func zalloc;
        free_func  zfree;
        voidpf     opaque;

        int     data_type;
        uLong   adler;
        uLong   reserved;
    } z_stream;

    typedef z_stream *z_streamp;

    typedef struct gz_header_s {
        int     text;
        uLong   time;
        int     xflags;
        int     os;
        Bytef   *extra;
        uInt    extra_len;
        uInt    extra_max;
        Bytef   *name;
        uInt    name_max;
        Bytef   *comment;
        uInt    comm_max;
        int     hcrc;
        int     done;
    } gz_header;

    typedef gz_header *gz_headerp;

    // zlib core C API signatures
    const char *zlibVersion();
    int deflateInit_(z_streamp strm, int level, const char *version, int stream_size);
    int deflate(z_streamp strm, int flush);
    int deflateEnd(z_streamp strm);
    
    int inflateInit_(z_streamp strm, const char *version, int stream_size);
    int inflate(z_streamp strm, int flush);
    int inflateEnd(z_streamp strm);
""")

# 2. Load the System zlib Library
if sys.platform == "win32":
    libz = ffi.dlopen("zlib1.dll")
elif sys.platform == "darwin":
    libz = ffi.dlopen("libz.dylib")
else:
    libz = ffi.dlopen("libz.so.1")

# Constants from zlib.h
Z_OK = 0
Z_STREAM_END = 1
Z_FINISH = 4
Z_DEFAULT_COMPRESSION = -1


class CFFIZlibEngine:
    """High-level Python wrapper executing native zlib routines via CFFI."""

    def __init__(self):
        self.version = libz.zlibVersion()
        self.version_str = ffi.string(self.version)

    def compress(self, data: bytes, level: int = Z_DEFAULT_COMPRESSION) -> bytes:
        """Compress raw bytes using native deflate via z_stream."""
        strm = ffi.new("z_stream *")
        
        # Macro expansion for deflateInit(strm, level)
        res = libz.deflateInit_(strm, level, self.version, ffi.sizeof("z_stream"))
        if res != Z_OK:
            raise RuntimeError(f"deflateInit failed with code {res}")

        try:
            in_buf = ffi.from_buffer(data)
            out_size = len(data) + 64  # Output buffer headroom
            out_buf = ffi.new("Bytef[]", out_size)

            strm.next_in = ffi.cast("const Bytef *", in_buf)
            strm.avail_in = len(data)
            strm.next_out = out_buf
            strm.avail_out = out_size

            res = libz.deflate(strm, Z_FINISH)
            if res != Z_STREAM_END:
                raise RuntimeError(f"deflate failed with code {res}")

            compressed_len = out_size - strm.avail_out
            return bytes(ffi.buffer(out_buf, compressed_len))

        finally:
            libz.deflateEnd(strm)

    def decompress(self, data: bytes, max_out_size: int = 1048576) -> bytes:
        """Decompress zlib stream using native inflate via z_stream."""
        strm = ffi.new("z_stream *")

        res = libz.inflateInit_(strm, self.version, ffi.sizeof("z_stream"))
        if res != Z_OK:
            raise RuntimeError(f"inflateInit failed with code {res}")

        try:
            in_buf = ffi.from_buffer(data)
            out_buf = ffi.new("Bytef[]", max_out_size)

            strm.next_in = ffi.cast("const Bytef *", in_buf)
            strm.avail_in = len(data)
            strm.next_out = out_buf
            strm.avail_out = max_out_size

            res = libz.inflate(strm, Z_FINISH)
            if res not in (Z_OK, Z_STREAM_END):
                raise RuntimeError(f"inflate failed with code {res}")

            decompressed_len = max_out_size - strm.avail_out
            return bytes(ffi.buffer(out_buf, decompressed_len))

        finally:
            libz.inflateEnd(strm)


if __name__ == "__main__":
    engine = CFFIZlibEngine()
    print(f"Loaded zlib C version: {engine.version_str.decode('utf-8')}")

    # Test Data
    payload = b"Hello from CFFI and zlib! " * 20
    print(f"Original size: {len(payload)} bytes")

    # Native Compression
    compressed = engine.compress(payload)
    print(f"Compressed size: {len(compressed)} bytes")

    # Native Decompression
    decompressed = engine.decompress(compressed)
    print(f"Decompressed size: {len(decompressed)} bytes")

    assert decompressed == payload, "Roundtrip data mismatch!"
    print("Verification succeeded!")
