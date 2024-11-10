import uvicorn
from fastapi import FastAPI
from api import ocr


app = FastAPI()
app.include_router(ocr.router)

@app.get("/")
def health_check():
    return {"ping": "pong"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860, reload=True)
