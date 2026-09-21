"""SQLHC HTML 补证解析：只识别 SQLHC，不把 ExaCheck/AWR/SQL Monitor 当成 SQLHC。"""

from __future__ import annotations

import unittest

from aiops_agent.application.sqlhc_report import parse_sqlhc_html


_SQLHC_HTML = """
<html>
  <head><title>SQLHC Report</title></head>
  <body>
    <h1>SQL Health-Check</h1>
    <p>SQL_ID: 6tjx7su0q5ttj</p>
    <table>
      <tr>
        <th>OWNER</th>
        <th>TABLE_NAME</th>
        <th>LAST_ANALYZED</th>
        <th>STALE_STATS</th>
      </tr>
      <tr>
        <td>APP</td>
        <td>ORDERS</td>
        <td>2024-01-01</td>
        <td>YES</td>
      </tr>
      <tr>
        <td>APP</td>
        <td>ITEMS</td>
        <td>2024-06-01</td>
        <td>NO</td>
      </tr>
    </table>
  </body>
</html>
"""

_EXACHECK_HTML = """
<html>
  <body>
    <h1>EXACHK Report</h1>
    <p>This host also has sqlhc.sql leftover.</p>
    <table>
      <tr><th>OWNER</th><th>TABLE_NAME</th><th>STALE_STATS</th></tr>
      <tr><td>APP</td><td>ORDERS</td><td>YES</td></tr>
    </table>
  </body>
</html>
"""

_AWR_HTML = """
<html>
  <body>
    <h1>AWR Report</h1>
    <table>
      <tr><th>Top Timed Events</th><td>db file sequential read</td></tr>
    </table>
  </body>
</html>
"""

_SQL_MONITOR_HTML = """
<html>
  <body>
    <h1>SQL Monitor Report</h1>
    <p>SQL_ID: 6tjx7su0q5ttj</p>
  </body>
</html>
"""


class SqlhcReportParseTests(unittest.TestCase):
    def test_sqlhc_html_extracts_stale_statistics(self) -> None:
        parsed = parse_sqlhc_html(
            _SQLHC_HTML,
            file_name="sqlhc_6tjx7su0q5ttj.html",
        )
        self.assertIsNotNone(parsed)
        self.assertEqual("6tjx7su0q5ttj", parsed.sql_id)
        self.assertEqual(
            (
                (
                    "6tjx7su0q5ttj",
                    "APP",
                    "ORDERS",
                    "TABLE",
                    "2024-01-01",
                    "YES",
                    "sqlhc_6tjx7su0q5ttj.html",
                ),
                (
                    "6tjx7su0q5ttj",
                    "APP",
                    "ITEMS",
                    "TABLE",
                    "2024-06-01",
                    "NO",
                    "sqlhc_6tjx7su0q5ttj.html",
                ),
            ),
            parsed.rows,
        )

    def test_exacheck_html_is_not_sqlhc(self) -> None:
        self.assertIsNone(
            parse_sqlhc_html(_EXACHECK_HTML, file_name="exachk.html")
        )

    def test_awr_html_is_not_sqlhc(self) -> None:
        self.assertIsNone(parse_sqlhc_html(_AWR_HTML, file_name="awr.html"))

    def test_sql_monitor_html_is_not_sqlhc(self) -> None:
        self.assertIsNone(
            parse_sqlhc_html(_SQL_MONITOR_HTML, file_name="sqlmon.html")
        )


if __name__ == "__main__":
    unittest.main()
