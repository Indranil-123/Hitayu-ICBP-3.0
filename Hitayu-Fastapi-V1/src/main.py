from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.PCOS_controller import pcos_router
from src.conversational_module.chat_Controller import cnv_router




# logger = setup_logger()

app = FastAPI(
    title ="Hitayu AI",
    description = "AI-Wellbeing",
    version = "1.0.0"
)

app.include_router(pcos_router)
app.include_router(cnv_router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    # logger.info("API is running")
    return {
        'status':'running',
        'service':'Hitayu AI',
        'version':'1.0.0'
    }
