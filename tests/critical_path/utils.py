import sqlite3
import tempfile


def create_test_db(dispatch_rows, symbol_rows):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    conn = sqlite3.connect(tmp.name)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE rocpd_kernel_dispatch_test (
            id INTEGER PRIMARY KEY,
            kernel_id INTEGER,
            queue_id INTEGER,
            start BIGINT,
            end BIGINT
        );
    """)

    cur.execute("""
        CREATE TABLE rocpd_info_kernel_symbol_test (
            id INTEGER PRIMARY KEY,
            display_name TEXT,
            kernel_name TEXT
        );
    """)

    cur.executemany(
        "INSERT INTO rocpd_kernel_dispatch_test VALUES (?, ?, ?, ?, ?);",
        dispatch_rows,
    )

    cur.executemany(
        "INSERT INTO rocpd_info_kernel_symbol_test VALUES (?, ?, ?);",
        symbol_rows,
    )

    conn.commit()
    conn.close()
    return tmp.name
