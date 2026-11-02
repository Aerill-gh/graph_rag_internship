import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from UI import streamlit_chat


class TestChatApiClient(unittest.TestCase):
    def test_sends_expected_request_and_returns_answer(self):
        response = io.BytesIO(json.dumps({
            "answer": "The policy is in the handbook. [S1]",
            "mode": "graphrag",
            "sources": [{
                "citation_id": "S1", "source_document": "employee_handbook.pdf",
                "chunk_id": "employee_handbook_002", "pages": [3, 4],
            }],
            "tools_used": ["explore_entity", "search_documents"],
        }).encode())
        with patch.object(streamlit_chat, "urlopen", return_value=response) as urlopen:
            result = streamlit_chat.chat_with_bot(
                "How often are reviews?", "http://127.0.0.1:8000/chat", mode="graphrag",
            )

        self.assertEqual(result.answer, "The policy is in the handbook. [S1]")
        self.assertEqual(result.mode, "graphrag")
        self.assertEqual(result.to_saved_text(), (
            "The policy is in the handbook. [S1]\n\nSources:\n"
            "- [S1] employee_handbook.pdf, pages 3, 4, chunk employee_handbook_002\n"
            "\nTools used: explore_entity → search_documents"
        ))
        request = urlopen.call_args.args[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(
            json.loads(request.data), {"message": "How often are reviews?", "mode": "graphrag"},
        )
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 120)

    def test_answer_without_sources_has_no_sources_block(self):
        response = io.BytesIO(b'{"answer": "I could not find enough information."}')
        with patch.object(streamlit_chat, "urlopen", return_value=response):
            result = streamlit_chat.chat_with_bot("Hi", "http://127.0.0.1:8000/chat")
        self.assertEqual(result.mode, "baseline")
        self.assertEqual(result.to_saved_text(), "I could not find enough information.")

    def test_history_for_api_drops_display_only_sources(self):
        messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer [S1]\n\nSources:\n- [S1] a.pdf", "mode": "baseline"},
        ]
        self.assertEqual(streamlit_chat.history_for_api(messages), [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer [S1]"},
        ])

    def test_blank_question_or_unknown_mode_does_not_send_request(self):
        with patch.object(streamlit_chat, "urlopen") as urlopen:
            with self.assertRaisesRegex(ValueError, "blank"):
                streamlit_chat.chat_with_bot("  ", "http://127.0.0.1:8000/chat")
            with self.assertRaisesRegex(ValueError, "mode"):
                streamlit_chat.chat_with_bot("Hi", "http://127.0.0.1:8000/chat", mode="other")
        urlopen.assert_not_called()

    def test_http_error_has_friendly_message(self):
        error = HTTPError(
            "http://127.0.0.1:8000/chat", 503, "unavailable", {}, io.BytesIO(b"secret detail"),
        )
        with patch.object(streamlit_chat, "urlopen", side_effect=error):
            with self.assertRaisesRegex(streamlit_chat.ChatServiceError, "HTTP 503"):
                streamlit_chat.chat_with_bot("Hello", "http://127.0.0.1:8000/chat")

    def test_http_error_shows_api_detail_sentence(self):
        body = json.dumps({"detail": "The graphrag agent is unavailable."}).encode()
        error = HTTPError("http://127.0.0.1:8000/chat", 503, "unavailable", {}, io.BytesIO(body))
        with patch.object(streamlit_chat, "urlopen", side_effect=error):
            with self.assertRaisesRegex(streamlit_chat.ChatServiceError, "graphrag agent is unavailable"):
                streamlit_chat.chat_with_bot("Hello", "http://127.0.0.1:8000/chat")

    def test_connection_error_has_friendly_message(self):
        with patch.object(
            streamlit_chat, "urlopen",
            side_effect=URLError("connection refused"),
        ):
            with self.assertRaisesRegex(streamlit_chat.ChatServiceError, "Cannot reach"):
                streamlit_chat.chat_with_bot("Hello", "http://127.0.0.1:8000/chat")

    def test_invalid_json_and_invalid_response_are_reported(self):
        for content in (b"not json", b'{"answer":""}', b"[]"):
            with self.subTest(content=content):
                with patch.object(streamlit_chat, "urlopen", return_value=io.BytesIO(content)):
                    with self.assertRaises(streamlit_chat.ChatServiceError):
                        streamlit_chat.chat_with_bot("Hello", "http://127.0.0.1:8000/chat")

    def test_invalid_endpoint_is_friendly_error(self):
        with self.assertRaisesRegex(streamlit_chat.ChatServiceError, "address is invalid"):
            streamlit_chat.chat_with_bot("Hello", "not a valid endpoint")

    def test_env_endpoint_and_default(self):
        with patch.object(streamlit_chat, "load_dotenv") as load_dotenv:
            with patch.dict(os.environ, {"CHATBOT_API_URL": "http://localhost:9000/chat"}):
                self.assertEqual(
                    streamlit_chat.get_chat_api_url(), "http://localhost:9000/chat",
                )
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(
                    streamlit_chat.get_chat_api_url(), streamlit_chat.DEFAULT_CHAT_API_URL,
                )
            self.assertEqual(load_dotenv.call_count, 2)


if __name__ == "__main__":
    unittest.main()
