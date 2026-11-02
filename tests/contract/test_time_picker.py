"""项目中所有纯时分输入统一使用共享双列选择器。"""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]


class TimePickerContractTest(unittest.TestCase):
    def test_business_pages_do_not_expose_native_time_inputs(self):
        for page in (ROOT / "ui").rglob("*.html"):
            self.assertIsNone(re.search(r'<input\b[^>]*type=["\']time["\']', page.read_text()), str(page))

    def test_inspection_uses_shared_picker_and_validates_before_cron(self):
        page = (ROOT / "ui/aiops/inspection-plans.html").read_text()
        script = (ROOT / "ui/aiops/js/aiops-inspection-plans.js").read_text()
        self.assertIn("../shared/time-picker.css", page)
        self.assertIn('type="module" src="./js/aiops-inspection-plans.js', page)
        self.assertIn('name="execution_time" type="text"', page)
        self.assertIn("../../shared/time-picker.js", script)
        self.assertIn("normalizeTime(form.elements.execution_time.value)", script)
        self.assertNotIn("showPicker", script)

    def test_picker_has_semantics_and_disposable_listeners(self):
        engine = (ROOT / "ui/shared/time-picker.js").read_text()
        for contract in ('role="listbox"', 'aria-activedescendant', 'aria-selected', 'aria-disabled', 'setCustomValidity', 'destroy()', 'observer.disconnect()', 'visualViewport'):
            self.assertIn(contract, engine)


if __name__ == "__main__":
    unittest.main()
