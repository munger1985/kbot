"""ExaCheck HTML 导入：只识别 ExaCheck/ORAchk，不把 SQLHC/AWR/SQL Monitor 当成 ExaCheck。"""

from __future__ import annotations

import unittest

from aiops_agent.application.exacheck_report import parse_exacheck_html
from aiops_agent.application.sqlhc_report import parse_sqlhc_html


_EXACHECK_HTML = """
<html>
  <head><title>EXACHK Report</title></head>
  <body>
    <h1>Oracle EXACHK</h1>
    <p>This host also has sqlhc.sql leftover.</p>
    <table>
      <tr>
        <th>Status</th>
        <th>Type</th>
        <th>Message</th>
        <th>Status On</th>
        <th>Check ID</th>
      </tr>
      <tr>
        <td>FAIL</td>
        <td>Hardware</td>
        <td>InfiniBand firmware is not current</td>
        <td>cel01</td>
        <td>IB_SWITCH_FW</td>
      </tr>
      <tr>
        <td>WARNING</td>
        <td>OS Check</td>
        <td>Database server RAM is below recommended</td>
        <td>db01</td>
        <td>OS_RAM</td>
      </tr>
      <tr>
        <td>INFO</td>
        <td>Software</td>
        <td>Clusterware version is 19.21</td>
        <td>db01</td>
        <td>CRS_VER</td>
      </tr>
      <tr>
        <td>PASS</td>
        <td>OS Check</td>
        <td>NTP is configured</td>
        <td>db01</td>
        <td>OS_NTP</td>
      </tr>
    </table>
  </body>
</html>
"""

_ORACHK_HTML = """
<html>
  <body>
    <h1>ORAchk Report</h1>
    <table>
      <tr>
        <th>Status</th>
        <th>Check Name</th>
        <th>Summary</th>
        <th>Hosts</th>
        <th>Check ID</th>
      </tr>
      <tr>
        <td>FAIL - Critical</td>
        <td>RMAN backup</td>
        <td>No successful backup in last 24 hours</td>
        <td>db01</td>
        <td>RMAN_BKP</td>
      </tr>
    </table>
  </body>
</html>
"""

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


class ExacheckReportParseTests(unittest.TestCase):
    def test_exacheck_html_extracts_fail_warning_and_info(self) -> None:
        parsed = parse_exacheck_html(
            _EXACHECK_HTML,
            file_name="exachk_db01.html",
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(
            (
                (
                    "FAIL",
                    "Hardware",
                    "InfiniBand firmware is not current",
                    "cel01",
                    "IB_SWITCH_FW",
                    "exachk_db01.html",
                ),
                (
                    "WARNING",
                    "OS Check",
                    "Database server RAM is below recommended",
                    "db01",
                    "OS_RAM",
                    "exachk_db01.html",
                ),
                (
                    "INFO",
                    "Software",
                    "Clusterware version is 19.21",
                    "db01",
                    "CRS_VER",
                    "exachk_db01.html",
                ),
            ),
            parsed.rows,
        )
        self.assertIsNone(parse_sqlhc_html(_EXACHECK_HTML, file_name="exachk_db01.html"))

    def test_orachk_html_uses_header_aliases(self) -> None:
        parsed = parse_exacheck_html(_ORACHK_HTML, file_name="orachk.html")
        self.assertIsNotNone(parsed)
        self.assertEqual(
            (
                (
                    "FAIL",
                    "RMAN backup",
                    "No successful backup in last 24 hours",
                    "db01",
                    "RMAN_BKP",
                    "orachk.html",
                ),
            ),
            parsed.rows,
        )

    def test_sqlhc_html_is_not_exacheck(self) -> None:
        self.assertIsNone(parse_exacheck_html(_SQLHC_HTML, file_name="sqlhc.html"))

    def test_awr_html_is_not_exacheck(self) -> None:
        self.assertIsNone(parse_exacheck_html(_AWR_HTML, file_name="awr.html"))

    def test_sql_monitor_html_is_not_exacheck(self) -> None:
        self.assertIsNone(
            parse_exacheck_html(_SQL_MONITOR_HTML, file_name="sqlmon.html")
        )


if __name__ == "__main__":
    unittest.main()
