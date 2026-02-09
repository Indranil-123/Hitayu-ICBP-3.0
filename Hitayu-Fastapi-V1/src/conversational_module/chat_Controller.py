from fastapi import FastAPI, WebSocket, APIRouter
from fastapi.responses import HTMLResponse
from .llm_service import LLM_response

cnv_router = APIRouter()

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'></ul>
        <script>
            var ws = new WebSocket("ws://localhost:8000/interact");
            
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""


@cnv_router.get("/chat")
async def get():
    return HTMLResponse(html)


@cnv_router.websocket("/interact")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            result = await LLM_response(data)  # Pydantic model returned
            await websocket.send_text(f"{result.reply} | intent: {result.intent}")
        except Exception as e:
            await websocket.send_text(f"Error: {str(e)}")