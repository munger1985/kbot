"""Agent Runtime Unicode 回复语言识别测试。"""

import unittest

from agent_runtime.language import (
    conversation_fallback_texts,
    detect_unicode_language,
    localized_message,
)


class LanguageDetectionTest(unittest.TestCase):
    def test_detects_supported_unicode_scripts(self):
        cases = (
            ("查一下 ChatBI 相关的 Asset", "zh-CN"),
            ("Find assets related to ChatBI", "en-US"),
            ("ChatBI 관련 자료를 찾아주세요", "ko-KR"),
            ("ChatBIに関連する資料を探してください", "ja-JP"),
            ("ابحث عن الأصول المتعلقة بالذكاء الاصطناعي", "ar"),
            ("Найдите материалы о ChatBI", "ru"),
        )

        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(expected, detect_unicode_language(text))

    def test_follow_up_without_letters_uses_recent_conversation_language(self):
        context = {
            "recent_items": [
                {"role": "USER", "content": {"text": "자산은 몇 개입니까?"}},
                {"role": "ASSISTANT", "content": {"text": "범위를 알려 주세요."}},
            ]
        }

        language = detect_unicode_language(
            "2?",
            fallback_texts=conversation_fallback_texts(context),
        )

        self.assertEqual("ko-KR", language)

    def test_defaults_to_english_when_no_script_is_available(self):
        self.assertEqual("en-US", detect_unicode_language("123 🎯"))

    def test_non_llm_fallback_uses_detected_language(self):
        message = localized_message("insufficient_evidence", "ar")

        self.assertTrue(message.startswith("لم يتم"))


if __name__ == "__main__":
    unittest.main()
