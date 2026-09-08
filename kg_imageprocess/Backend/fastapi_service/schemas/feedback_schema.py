from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    user_id: str = Field(..., description="شناسه یکتای کاربر (UUID از جانگو)")
    session_id: Optional[str] = Field(None, description="شناسه جلسه تمرین اخیر (در صورت وجود)")
    message: str = Field(..., description="سوال یا پیام متنی/کلامی کاربر")

class MemoryFact(BaseModel):
    fact: str
    valid_at: Optional[str] = None

class ChatResponse(BaseModel):
    user_id: str
    reply: str = Field(..., description="پاسخ روان و مربی‌گونه هوش مصنوعی")
    retrieved_facts: List[MemoryFact] = Field(default_factory=list, description="حقایق استخراج‌شده از گراف زمان‌مند")