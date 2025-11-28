# main.py  (located in project root)

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "deployment.app:app",   # module:variable
        host="127.0.0.1",
        port=8000,
        reload=True
    )
