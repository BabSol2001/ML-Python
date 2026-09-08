from fastapi import APIRouter, HTTPException, status
from schemas.feedback_schema import ChatRequest, ChatResponse
from services.rag_service import rag_service

router = APIRouter(prefix="/api/v1/chat", tags=["Post-Workout Coaching"])

@router.post("/feedback", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def get_post_workout_feedback(request: ChatRequest):
    """
    اندپوینت دریافت تحلیل عمیق مربیگری پس از تمرین با استفاده از MS-RAG و Graphiti
    """
    try:
        reply, retrieved_facts = await rag_service.generate_coaching_feedback(
            user_id=request.user_id,
            user_message=request.message
        )

        return ChatResponse(
            user_id=request.user_id,
            reply=reply,
            retrieved_facts=retrieved_facts
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"خطا در تولید پاسخ RAG: {str(e)}"
        )