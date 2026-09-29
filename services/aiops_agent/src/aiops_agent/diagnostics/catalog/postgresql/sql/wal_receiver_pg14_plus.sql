SELECT
    pid,
    status,
    CAST(receive_start_lsn AS text) AS receive_start_lsn,
    receive_start_tli,
    CAST(written_lsn AS text) AS received_lsn,
    received_tli,
    last_msg_send_time,
    last_msg_receipt_time,
    CAST(latest_end_lsn AS text) AS latest_end_lsn,
    latest_end_time,
    slot_name,
    sender_host,
    sender_port
FROM pg_stat_wal_receiver
