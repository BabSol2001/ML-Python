from rest_framework import viewsets, status
from rest_framework.views import APIView
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

from .models import User, BiomechanicalSession, FrameAnalysis
from .serializers import (
    AthleteContextSerializer,
    BiomechanicalSessionSerializer,
    FrameAnalysisSerializer
)


class InternalAthleteContextView(APIView):
    """
    اندپوینت اختصاصی دریافت Context ورزشکار برای سرویس FastAPI و آنتولوژی
    """
    def get(self, request, user_id):
        user = get_object_or_404(User, id=user_id)
        serializer = AthleteContextSerializer(user)
        return Response(serializer.data, status=status.HTTP_200_OK)


class BiomechanicalSessionViewSet(viewsets.ModelViewSet):
    """
    ایجاد و مدیریت نشست‌های تحلیل بیومکانیکی
    """
    queryset = BiomechanicalSession.objects.all()
    serializer_class = BiomechanicalSessionSerializer


class FrameAnalysisViewSet(viewsets.ModelViewSet):
    """
    ثبت و دریافت خروجی فریم‌های تحلیل‌شده توسط خط لوله FastAPI
    """
    queryset = FrameAnalysis.objects.all()
    serializer_class = FrameAnalysisSerializer