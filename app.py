from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from task_handler import parse_task_with_llm, execute_task_function

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/run")
async def run_task(task: str = Query(...)):
    try:
        # Use LLM to parse the task description
        function_name, parameters = parse_task_with_llm(task)
        
        # Execute the corresponding task function with additional arguments
        result = execute_task_function(function_name, **parameters)
        return {"status": "success", "details": result}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/read")
    
async def read_file(path):
    try:
        with open(path, "r",  encoding="utf-8") as file:
            content =  file.read()
        return PlainTextResponse(content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

