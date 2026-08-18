"""Agent Runtime Unicode 回复语言识别测试。"""

import unittest

from agent_runtime.language import (
    answer_matches_language,
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

    def test_rejects_chinese_explanation_for_english_question(self):
        answer = (
            "根据知识库中的证据，有一个资产与 ChatBI 相关："
            "**Conversational Banking with Select AI Agents** [C1]"
        )

        self.assertFalse(answer_matches_language(answer, "en-US"))

    def test_accepts_english_explanation_with_source_title(self):
        answer = (
            "One asset is related to ChatBI: "
            "**Conversational Banking with Select AI Agents** [C1]"
        )

        self.assertTrue(answer_matches_language(answer, "en-US"))

    def test_accepts_chinese_explanation_with_english_product_name(self):
        answer = "知识库中有一个与 ChatBI 相关的 Asset。[C1]"

        self.assertTrue(answer_matches_language(answer, "zh-CN"))

    def test_accepts_chinese_asset_list_with_english_metadata(self):
        answer = (
            "找到以下与 ChatBI 相关的 Asset：\n"
            "- Deep Data Security with IAM in Agentic Application Demo\n"
            "  作者：HYSUN.HE@ORACLE.COM\n"
            "  主题：ChatBI, AI / Machine Learning, RAG, Security Solution "
            "[C1]"
        )

        self.assertTrue(answer_matches_language(answer, "zh-CN"))

    def test_rejects_english_answer_with_isolated_chinese_term(self):
        answer = "One asset relates to 数据库 and ChatBI. [C1]"

        self.assertFalse(answer_matches_language(answer, "zh-CN"))

    def test_ignores_unformatted_source_title(self):
        answer = "One related source is 数据库智能运维与故障诊断实践。[C1]"

        self.assertTrue(
            answer_matches_language(
                answer,
                "en-US",
                ignored_texts=("数据库智能运维与故障诊断实践",),
            )
        )


if __name__ == "__main__":
    unittest.main()
