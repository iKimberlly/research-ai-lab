from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.rag.pipeline import RAGPipeline


router = APIRouter(
    prefix="/api/chat",
    tags=["Chat"]
)


# ============================================================
# REQUEST
# ============================================================

class ChatRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=3,
        description="Pergunta sobre os documentos"
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Quantidade de chunks recuperados"
    )


# ============================================================
# RESPONSE
# ============================================================

class Source(BaseModel):

    chunk: int

    source: str

    distance: float


class ChatResponse(BaseModel):

    question: str

    answer: str

    sources: list[Source]


# ============================================================
# ENDPOINT
# ============================================================

@router.post(
    "",
    response_model=ChatResponse
)
def chat(
    request: ChatRequest
):

    try:

        pipeline = RAGPipeline(
            top_k=request.top_k
        )

        result = pipeline.ask(
            request.question
        )

        return result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )