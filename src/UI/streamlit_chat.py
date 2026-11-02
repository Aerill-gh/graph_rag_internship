import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHAT_API_URL = "http://127.0.0.1:8000/chat"
DEFAULT_BACKEND_API_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SECONDS = 120
SOURCES_SEPARATOR = "\n\nSources:\n"
# Labels shown in the sidebar switch and on each answer.
AGENT_MODES = {"baseline": "Baseline", "graphrag": "GraphRAG"}
MAX_TITLE_LENGTH = 60
MAX_PINNED_CHATS = 3
logger = logging.getLogger(__name__)


class ChatServiceError(RuntimeError):
    """A safe, user-displayable error from the chatbot API."""


@dataclass(frozen=True)
class ChatAnswer:
    answer: str
    mode: str
    sources: list[dict] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)

    def to_saved_text(self) -> str:
        """Answer text plus a Sources block, in the format _render_answer displays."""
        lines = []
        for source in self.sources:
            pages = ", ".join(str(page) for page in source.get("pages") or []) or "unknown"
            lines.append(
                f"- [{source.get('citation_id', '?')}] {source.get('source_document', '?')}, "
                f"pages {pages}, chunk {source.get('chunk_id', '?')}"
            )
        if self.tools_used:
            # The tool list shows whether GraphRAG really used the graph for this answer.
            lines.append(f"\nTools used: {' → '.join(self.tools_used)}")
        return self.answer + (SOURCES_SEPARATOR + "\n".join(lines) if lines else "")


@dataclass(frozen=True)
class Conversation:
    """Minimal conversation info from backend."""
    id: str
    title: str
    pinned: bool


def get_chat_api_url() -> str:
    """Load the project .env and return the configured chatbot API endpoint."""
    load_dotenv(PROJECT_ROOT / ".env")
    return os.getenv("CHATBOT_API_URL", DEFAULT_CHAT_API_URL).strip() or DEFAULT_CHAT_API_URL


def get_backend_api_url() -> str:
    """Load the project .env and return the configured backend API endpoint."""
    load_dotenv(PROJECT_ROOT / ".env")
    return os.getenv("BACKEND_API_URL", DEFAULT_BACKEND_API_URL).strip() or DEFAULT_BACKEND_API_URL


def _make_backend_request(
    method: str, endpoint: str, data: dict | None = None, params: dict | None = None
) -> dict:
    """Helper to make HTTP requests to backend API endpoints."""
    url = get_backend_api_url() + endpoint
    if params:
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{query_string}"
    
    body = json.dumps(data).encode("utf-8") if data else None
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method=method,
    )
    
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = _error_detail(exc)
        exc.close()
        logger.error("Backend API returned HTTP %s.", exc.code)
        raise ChatServiceError(
            detail or f"The chatbot service returned HTTP {exc.code}. Check that it is running."
        ) from None
    except (TimeoutError, URLError, OSError) as exc:
        logger.error("Backend API request failed (%s).", type(exc).__name__)
        raise ChatServiceError(
            "Cannot reach the chatbot service. Make sure the chatbot API is running, "
            "then try again."
        ) from None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.error("Backend API returned an invalid response (%s).", type(exc).__name__)
        raise ChatServiceError(
            "The chatbot service returned an invalid response. Please try again."
        ) from None


def list_conversations() -> list[Conversation]:
    """Fetch all conversations from the backend."""
    try:
        result = _make_backend_request("GET", "/conversations")
        return [
            Conversation(id=item["id"], title=item["title"], pinned=item["pinned"])
            for item in result
        ]
    except ChatServiceError:
        raise
    except Exception as exc:
        logger.error("Failed to fetch conversations (%s).", type(exc).__name__)
        raise ChatServiceError("Cannot load chat history. Please try again.") from None


def create_conversation() -> Conversation:
    """Create a new conversation on the backend."""
    try:
        result = _make_backend_request("POST", "/conversations")
        return Conversation(id=result["id"], title=result["title"], pinned=result["pinned"])
    except ChatServiceError:
        raise
    except Exception as exc:
        logger.error("Failed to create conversation (%s).", type(exc).__name__)
        raise ChatServiceError("Cannot create a new chat. Please try again.") from None


def get_conversation_messages(conversation_id: str) -> list[dict]:
    """Fetch all messages for a conversation from the backend."""
    try:
        result = _make_backend_request("GET", f"/conversations/{conversation_id}/messages")
        return result.get("messages", [])
    except ChatServiceError:
        raise
    except Exception as exc:
        logger.error("Failed to fetch messages (%s).", type(exc).__name__)
        raise ChatServiceError("Cannot load chat messages. Please try again.") from None


def rename_conversation(conversation_id: str, title: str) -> Conversation:
    """Rename a conversation on the backend."""
    try:
        result = _make_backend_request(
            "PATCH", f"/conversations/{conversation_id}", data={"title": title}
        )
        return Conversation(id=result["id"], title=result["title"], pinned=result["pinned"])
    except ChatServiceError:
        raise
    except Exception as exc:
        logger.error("Failed to rename conversation (%s).", type(exc).__name__)
        raise ChatServiceError("Cannot rename chat. Please try again.") from None


def set_conversation_pinned(conversation_id: str, pinned: bool) -> Conversation:
    """Update the pinned status of a conversation on the backend."""
    try:
        result = _make_backend_request(
            "PATCH", f"/conversations/{conversation_id}", data={"pinned": pinned}
        )
        return Conversation(id=result["id"], title=result["title"], pinned=result["pinned"])
    except ChatServiceError:
        raise
    except Exception as exc:
        logger.error("Failed to update conversation (%s).", type(exc).__name__)
        raise ChatServiceError("Cannot update chat. Please try again.") from None


def delete_conversation(conversation_id: str) -> None:
    """Delete a conversation on the backend."""
    try:
        _make_backend_request("DELETE", f"/conversations/{conversation_id}")
    except ChatServiceError:
        raise
    except Exception as exc:
        logger.error("Failed to delete conversation (%s).", type(exc).__name__)
        raise ChatServiceError("Cannot delete chat. Please try again.") from None


def history_for_api(messages: list[dict]) -> list[dict[str, str]]:
    """Send only role and answer text: the old Sources blocks are display-only."""
    return [
        {"role": message["role"], "content": message["content"].split(SOURCES_SEPARATOR, 1)[0]}
        for message in messages
    ]


def chat_with_bot(
    question: str,
    endpoint: str | None = None,
    *,
    history: list[dict[str, str]] | None = None,
    mode: str = "baseline",
    conversation_id: str | None = None,
) -> ChatAnswer:
    """Send one question to the chatbot API and return its answer."""
    if not question.strip():
        raise ValueError("Question cannot be blank.")
    if mode not in AGENT_MODES:
        raise ValueError(f"Unknown agent mode: {mode}")

    body: dict[str, object] = {"message": question, "mode": mode}
    if history is not None:
        body["history"] = history
    if conversation_id is not None:
        body["conversation_id"] = conversation_id
    try:
        request = Request(
            endpoint or get_chat_api_url(),
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
    except ValueError:
        logger.error("Chat API endpoint is invalid.")
        raise ChatServiceError(
            "The chatbot API address is invalid. Check CHATBOT_API_URL in .env."
        ) from None

    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = _error_detail(exc)
        exc.close()
        logger.error("Chat API returned HTTP %s.", exc.code)
        raise ChatServiceError(
            detail or f"The chatbot service returned HTTP {exc.code}. Check that it is running."
        ) from None
    except (TimeoutError, URLError, OSError) as exc:
        logger.error("Chat API request failed (%s).", type(exc).__name__)
        raise ChatServiceError(
            "Cannot reach the chatbot service. Make sure the chatbot API is running, "
            "then try again."
        ) from None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.error("Chat API returned an invalid response (%s).", type(exc).__name__)
        raise ChatServiceError(
            "The chatbot service returned an invalid response. Please try again."
        ) from None

    answer = payload.get("answer") if isinstance(payload, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        logger.error("Chat API response is missing a non-empty answer.")
        raise ChatServiceError(
            "The chatbot service returned an empty or invalid answer. Please try again."
        )
    sources = payload.get("sources")
    tools_used = payload.get("tools_used")
    return ChatAnswer(
        answer=answer,
        mode=payload.get("mode") if payload.get("mode") in AGENT_MODES else mode,
        sources=[s for s in sources if isinstance(s, dict)] if isinstance(sources, list) else [],
        tools_used=[t for t in tools_used if isinstance(t, str)]
        if isinstance(tools_used, list) else [],
    )


def _error_detail(exc: HTTPError) -> str | None:
    """Return the API's own readable error text (FastAPI's "detail"), if there is one."""
    try:
        detail = json.loads(exc.read().decode("utf-8")).get("detail")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        return None
    # Validation errors (422) have a list here; only plain sentences are shown to users.
    return detail if isinstance(detail, str) and detail.strip() else None


def _render_answer(answer: str, mode: str | None = None) -> None:
    """Show which agent answered, the answer, and its sources in an expandable section."""
    if mode in AGENT_MODES:
        st.caption(f":material/smart_toy: {AGENT_MODES[mode]}")
    parts = answer.split(SOURCES_SEPARATOR, 1)
    st.markdown(parts[0])
    if len(parts) == 2:
        with st.expander("Sources"):
            st.markdown(parts[1])


def _dismiss_chat_dialog() -> None:
    st.session_state.pop("chat_dialog", None)


@st.dialog("Rename chat", on_dismiss=_dismiss_chat_dialog)
def _rename_chat_dialog(chat: Conversation) -> None:
    with st.form(f"rename_form_{chat.id}"):
        name = st.text_input("Chat name", value=chat.title, max_chars=MAX_TITLE_LENGTH)
        save = st.form_submit_button("Save", use_container_width=True)
    if save:
        try:
            rename_conversation(chat.id, name)
        except ChatServiceError as exc:
            st.error(str(exc))
        else:
            _dismiss_chat_dialog()
            st.rerun()
    if st.button("Cancel", key=f"cancel_rename_{chat.id}", use_container_width=True):
        _dismiss_chat_dialog()
        st.rerun()


@st.dialog("Delete chat?", on_dismiss=_dismiss_chat_dialog)
def _delete_chat_dialog(chat: Conversation) -> None:
    st.write(f'Delete "{chat.title}" and its saved messages? This cannot be undone.')
    cancel, delete = st.columns(2)
    if cancel.button("Cancel", key=f"cancel_delete_{chat.id}", use_container_width=True):
        _dismiss_chat_dialog()
        st.rerun()
    if delete.button(
        "Delete", key=f"confirm_delete_{chat.id}", type="primary", use_container_width=True,
    ):
        try:
            delete_conversation(chat.id)
        except ChatServiceError as exc:
            st.error(str(exc))
        else:
            _dismiss_chat_dialog()
            if chat.id == st.session_state.get("active_chat_id"):
                st.session_state.pop("active_chat_id", None)
                st.session_state.pop("pending_prompt", None)
            st.rerun()


def _render_chat_list(
    chats: list[Conversation], active_id: str,
) -> tuple[str, Conversation] | None:
    action = None
    if st.button(
        "New chat", icon=":material/edit_square:", key="new_chat", use_container_width=True,
    ):
        try:
            new_chat = create_conversation()
            st.session_state.active_chat_id = new_chat.id
        except ChatServiceError as exc:
            st.error(str(exc))
        st.session_state.pop("pending_prompt", None)
        st.rerun()

    pin_count = sum(chat.pinned for chat in chats)
    st.caption("Chats")
    for chat in chats:
        row_key = "chat_row_active" if chat.id == active_id else "chat_row"
        with st.container(key=f"{row_key}_{chat.id}"):
            select, actions = st.columns([8, 1], gap="small", vertical_alignment="center")
            if select.button(
                chat.title,
                icon=":material/keep:" if chat.pinned else None,
                key=f"select_{chat.id}",
                help=chat.title,
                use_container_width=True,
            ):
                st.session_state.active_chat_id = chat.id
                st.session_state.pop("pending_prompt", None)
                st.rerun()
            with actions.popover("⋯", use_container_width=True):
                if st.button(
                    "Unpin" if chat.pinned else "Pin",
                    icon=":material/keep_off:" if chat.pinned else ":material/keep:",
                    key=f"pin_{chat.id}",
                    disabled=not chat.pinned and pin_count >= MAX_PINNED_CHATS,
                    help="Maximum three pinned chats. Unpin one first."
                    if not chat.pinned and pin_count >= MAX_PINNED_CHATS else None,
                    use_container_width=True,
                ):
                    try:
                        set_conversation_pinned(chat.id, not chat.pinned)
                    except ChatServiceError as exc:
                        st.error(str(exc))
                    else:
                        st.rerun()
                if st.button(
                    "Rename", icon=":material/edit:", key=f"rename_{chat.id}",
                    use_container_width=True,
                ):
                    action = ("rename", chat)
                if st.button(
                    "Delete", icon=":material/delete:", key=f"delete_{chat.id}",
                    use_container_width=True,
                ):
                    action = ("delete", chat)
    return action


def _storage_error() -> None:
    logger.exception("Backend chat service is unavailable.")
    st.error("Cannot reach the chat backend service. Ensure it's running on the configured URL.")


def main() -> None:
    try:
        _main()
    except ChatServiceError:
        _storage_error()
        st.stop()


def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    st.set_page_config(
        page_title="Zava Assistant",
        page_icon="💬",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background-image: linear-gradient(
                180deg, rgba(8, 126, 130, .08) 0, transparent 19rem
            );
        }
        [data-testid="stHeader"] { background: transparent; }
        /* Streamlit 1.66 menu actions; keep theme, cache and print controls. */
        [data-testid="stMainMenuItem-rerun"],
        [data-testid="stMainMenuItem-autoRerun"],
        [data-testid="stMainMenuItem-recordScreencast"] { display: none; }
        .block-container { max-width: 900px; padding-top: 2rem; padding-bottom: 6rem; }
        .zava-kicker { color: color-mix(in srgb, currentColor 55%, #087e82);
                       font-size: .78rem; font-weight: 700;
                       letter-spacing: .13em; text-transform: uppercase; }
        .zava-subtitle { color: inherit; opacity: .8;
                         font-size: 1.06rem; margin-top: -.5rem; }
        [data-testid="stChatMessage"] {
            border: 1px solid color-mix(in srgb, currentColor 16%, transparent);
            border-radius: 16px;
                                        padding: 1rem 1.1rem; margin-bottom: .9rem; }
        [data-testid="stChatInput"] { border-radius: 16px; }
        .stButton > button { border-radius: 12px; }
        [data-testid="stSidebarHeader"] {
            height: 0; min-height: 0; margin-bottom: 0; overflow: visible;
        }
        [data-testid="stSidebarCollapseButton"] {
            position: absolute; top: .5rem; right: .5rem; z-index: 1;
        }
        [data-testid="stSidebarUserContent"] { padding-top: 0; }
        [data-testid="stSidebar"] .st-key-sidebar_logo {
            transform: translateX(-.4rem);
        }
        [data-testid="stSidebar"] .st-key-new_chat button {
            justify-content: flex-start; background: transparent; border: 0;
        }
        [data-testid="stSidebar"] .st-key-new_chat button > div,
        [data-testid="stSidebar"] [class*="st-key-select_"] button > div {
            width: 100%; justify-content: flex-start; min-width: 0;
        }
        [data-testid="stSidebar"] [class*="st-key-chat_row"] {
            border-radius: 12px; padding: 0 .35rem;
        }
        [data-testid="stSidebar"] [class*="st-key-chat_row"]:hover,
        [data-testid="stSidebar"] [class*="st-key-chat_row_active"] {
            background: rgba(128, 128, 128, .12);
        }
        [data-testid="stSidebar"] [class*="st-key-chat_row"] button {
            background: transparent; border: 0; box-shadow: none;
            min-height: 2.7rem; padding: .35rem .25rem;
        }
        [data-testid="stSidebar"] [class*="st-key-chat_row"]
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {
            flex: 1 1 0; min-width: 0;
        }
        [data-testid="stSidebar"] [class*="st-key-chat_row"]
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {
            flex: 0 0 2.5rem; min-width: 2.5rem;
        }
        [data-testid="stSidebar"] [class*="st-key-select_"] button {
            justify-content: flex-start;
        }
        [data-testid="stSidebar"] [class*="st-key-select_"] button p {
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
            text-align: left;
        }
        [data-testid="stSidebar"] [class*="st-key-chat_row"]
        [data-testid="stPopoverButton"] > div > [aria-hidden="true"] {
            display: none;
        }
        [data-testid="stSidebar"] [class*="st-key-chat_row"]
        [data-testid="stPopoverButton"] {
            width: 2.5rem; min-width: 2.5rem; color: inherit;
        }
        [data-testid="stPopoverBody"]:has([class*="st-key-pin_"]) {
            min-width: 180px; padding: .5rem;
        }
        [data-testid="stPopoverBody"]:has([class*="st-key-pin_"])
        [data-testid="stVerticalBlock"] { gap: .15rem; }
        [data-testid="stPopoverBody"] [class*="st-key-pin_"] button,
        [data-testid="stPopoverBody"] [class*="st-key-rename_"] button,
        [data-testid="stPopoverBody"] [class*="st-key-delete_"] button {
            border: 0; background: transparent; justify-content: flex-start;
        }
        [data-testid="stPopoverBody"] [class*="st-key-delete_"] button {
            color: color-mix(in srgb, currentColor 20%, #ef4444);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Load conversations from backend
    try:
        chats = list_conversations()
    except ChatServiceError:
        _storage_error()
        st.stop()
    
    if not chats:
        try:
            new_chat = create_conversation()
            chats = [new_chat]
        except ChatServiceError:
            _storage_error()
            st.stop()
    
    if st.session_state.get("active_chat_id") not in {chat.id for chat in chats}:
        st.session_state.active_chat_id = chats[0].id
    active_id = st.session_state.active_chat_id
    active = next(chat for chat in chats if chat.id == active_id)
    
    try:
        messages = get_conversation_messages(active_id)
    except ChatServiceError:
        _storage_error()
        st.stop()

    with st.sidebar:
        with st.container(key="sidebar_logo"):
            st.image(str(PROJECT_ROOT / "UI" / "assets" / "zava_logo.png"), width="stretch")
        st.caption("Ask questions about company roles, policies, and benefits.")
        # Same agent and instructions in both modes; GraphRAG only adds graph tools.
        mode = st.radio(
            "Agent",
            options=list(AGENT_MODES),
            format_func=AGENT_MODES.get,
            horizontal=True,
            key="agent_mode",
            captions=["Document search only", "Documents + knowledge graph"],
        )
        action = _render_chat_list(chats, active_id)
        st.divider()
        st.caption("Answers may be incomplete. Check the cited source passages.")

    if action is not None:
        kind, chat = action
        st.session_state.chat_dialog = (kind, chat.id)
    if dialog := st.session_state.get("chat_dialog"):
        kind, chat_id = dialog
        dialog_chat = next((chat for chat in chats if chat.id == chat_id), None)
        if dialog_chat is None:
            _dismiss_chat_dialog()
        elif kind == "rename":
            _rename_chat_dialog(dialog_chat)
        else:
            _delete_chat_dialog(dialog_chat)

    st.markdown('<div class="zava-kicker">Zava · Employee knowledge assistant</div>',
                unsafe_allow_html=True)
    st.title(active.title if messages else "How can I help?")
    st.markdown(
        '<p class="zava-subtitle">Ask about Zava roles, workplace policies, '
        "and employee benefits.</p>",
        unsafe_allow_html=True,
    )

    for message in messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_answer(message["content"], message.get("mode"))
            else:
                st.markdown(message["content"])

    if error := st.session_state.pop("chat_error", None):
        st.error(error)

    if not messages:
        st.markdown("#### Try a question")
        suggestions = [
            "How often are performance reviews conducted?",
            "What does Northwind Health Plus cover?",
            "What qualifications does the CFO role require?",
        ]
        columns = st.columns(len(suggestions))
        for column, suggestion in zip(columns, suggestions):
            if column.button(suggestion, use_container_width=True, key=suggestion):
                st.session_state.pending_prompt = suggestion

    prompt = st.chat_input("Ask a question about Zava...")
    prompt = st.session_state.pop("pending_prompt", None) or prompt
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            spinner = (
                "Searching documents and the knowledge graph..." if mode == "graphrag"
                else "Searching Zava documents and preparing an answer..."
            )
            with st.spinner(spinner):
                try:
                    result = chat_with_bot(
                        prompt, 
                        history=history_for_api(messages), 
                        mode=mode,
                        conversation_id=active_id,
                    )
                except ChatServiceError as exc:
                    st.session_state.chat_error = str(exc)
                else:
                    _render_answer(result.to_saved_text(), result.mode)
        st.rerun()


if __name__ == "__main__":
    main()
