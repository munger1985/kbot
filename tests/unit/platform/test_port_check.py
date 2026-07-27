"""微服务端口占用检查回归测试。"""

import errno
import unittest
from unittest.mock import MagicMock, patch

from platform_core.platform.port_check import check_port_available


class PortCheckTest(unittest.TestCase):
    def test_address_in_use_returns_false_without_attribute_error(self):
        fake_socket = MagicMock()
        fake_socket.__enter__.return_value.bind.side_effect = OSError(
            errno.EADDRINUSE,
            "Address already in use",
        )
        with (
            patch(
                "platform_core.platform.port_check.socket.socket",
                return_value=fake_socket,
            ),
            patch("platform_core.platform.port_check.logger.error"),
            patch("platform_core.platform.port_check.print"),
        ):
            self.assertFalse(
                check_port_available("127.0.0.1", 18090, "kc-api")
            )


if __name__ == "__main__":
    unittest.main()
