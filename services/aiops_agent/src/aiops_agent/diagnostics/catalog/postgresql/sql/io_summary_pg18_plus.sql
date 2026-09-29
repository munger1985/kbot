SELECT
    backend_type,
    object AS object_type,
    context AS io_context,
    reads,
    read_bytes,
    read_time AS read_time_ms,
    writes,
    write_bytes,
    write_time AS write_time_ms,
    writebacks,
    writeback_time AS writeback_time_ms,
    extends,
    extend_bytes,
    extend_time AS extend_time_ms,
    hits,
    evictions,
    reuses,
    fsyncs,
    fsync_time AS fsync_time_ms,
    stats_reset
FROM pg_stat_io
ORDER BY backend_type, object, context
