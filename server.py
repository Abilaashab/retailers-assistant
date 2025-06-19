from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import process_enhanced_query
import uvicorn
import json
from typing import Dict, Any, Optional

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    skip_translation: bool = False
    stream: bool = False

def is_english(text: str) -> bool:
    """Check if the text is in English."""
    # Simple check - could be enhanced with a more robust solution
    try:
        return text.encode('ascii').decode('ascii') == text
    except UnicodeEncodeError:
        return False

@app.post("/process_query")
async def process_query(request: QueryRequest):
    try:
        query = request.query
        
        # Skip translation if requested or if the text is already in English
        if not request.skip_translation and not is_english(query):
            # Add translation logic here if needed
            pass
            
        # Process the query using your existing agent
        result = process_enhanced_query(query)
        
        # If streaming is requested, return a streaming response
        if request.stream:
            from fastapi.responses import StreamingResponse
            import asyncio
            
            async def stream_response():
                # Split the result into words to simulate streaming
                words = result.split()
                for word in words:
                    # Send each word as a separate chunk
                    yield json.dumps({"response": f"{word} "}) + "\n"
                    await asyncio.sleep(0.05)  # Small delay between chunks
            
            return StreamingResponse(stream_response(), media_type="text/event-stream")
        
        # Return the response in the expected format
        return {
            "response": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
