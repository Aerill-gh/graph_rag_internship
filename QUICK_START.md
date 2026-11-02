# Quick Start: Chat History Refactoring

## What Changed?
- ✅ Chat history moved from UI to backend
- ✅ UI is now stateless (only displays data)
- ✅ Backend owns all persistence

## What You Need to Do

### 1. Kill Old Backend Process (if stuck)
```powershell
# Find what's on port 8000
netstat -ano | findstr "8000"

# If process 6556 is stuck, restart your terminal or machine
# Then the port will be free
```

### 2. Start Backend
```powershell
cd C:\Users\ihor.holdanskyi\graph_rag_internship
uv run python -m chatbot.chat_bot_main
```

Expected output:
```
INFO:     Started server process [XXXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### 3. Start UI (in new terminal)
```powershell
cd C:\Users\ihor.holdanskyi\graph_rag_internship
uv run streamlit run src/UI/streamlit_chat.py
```

Expected output:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

### 4. Test in Browser
Visit `http://localhost:8501`

**Test these operations:**
- ✅ Create a new chat
- ✅ Ask a question
- ✅ Refresh the page → messages should still be there
- ✅ Rename a chat
- ✅ Pin/unpin a chat
- ✅ Delete a chat

## What's New

### Backend Endpoints
```
GET    /conversations                    → List all chats
POST   /conversations                    → Create new chat
GET    /conversations/{id}/messages      → Get chat messages
PATCH  /conversations/{id}               → Rename or pin/unpin
DELETE /conversations/{id}               → Delete chat
POST   /chat                             → Ask question (auto-saves)
```

### Environment Variable
Add to `.env`:
```
BACKEND_API_URL=http://127.0.0.1:8000
```
(Already defaults to this if not set)

## Test Results
```
Backend Tests:  5/5 PASS ✅
UI Tests:      10/10 PASS ✅
Total:         15/15 PASS ✅
```

## Git Commits
```
3fb2cc7 - Refactor: Move chat history management to backend API
f2f5b5f - Add conversation management endpoints to backend
```

## If Something Goes Wrong

**Error: "Cannot reach the chat backend service"**
- Backend is not running
- Start backend: `uv run python -m chatbot.chat_bot_main`

**Error: "Port 8000 already in use"**
- Old process still running
- Check: `netstat -ano | findstr "8000"`
- Restart your machine or close the old process

**Tests failing**
- Run: `uv run python -m unittest discover tests -v`
- All 15 tests should pass

## Documentation
See [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md) for full details.

---

**Everything is implemented, tested, and committed!** 🎉

You're ready to test the full system.
