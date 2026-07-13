from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "message": "This repo hosts a Streamlit app. Run it locally with `streamlit run app.py`.",
        "github": "https://github.com/ramdhote05/Enterprise-knowledge-assistant",
    }
