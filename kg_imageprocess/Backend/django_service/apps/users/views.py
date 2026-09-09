from rest_framework import viewsets, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.serializers import BaseSerializer
from django.shortcuts import get_object_or_404

from .models import User, UserProfile, BiomechanicalSession, FrameAnalysis
from .serializers import (
    UserCreateSerializer,
    AthleteContextSerializer,
    UserProfileSerializer,
    BiomechanicalSessionSerializer,
    FrameAnalysisSerializer
)


class UserViewSet(viewsets.ModelViewSet):
    """
    ایجاد (ثبت‌نام)، مشاهده و مدیریت کامل کاربران
    """
    queryset = User.objects.all()
    serializer_class = UserCreateSerializer

    def get_serializer_class(self) -> type[BaseSerializer]: # type: ignore[override]
        if self.action in ['retrieve', 'list']:
            return AthleteContextSerializer
        return UserCreateSerializer


class UserProfileViewSet(viewsets.ModelViewSet):
    """
    مشاهده و ویرایش پروفایل‌های کاربران بر اساس user_id
    """
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer
    lookup_field = 'user_id'


class InternalAthleteContextView(APIView):
    """
    اندپوینت اختصاصی دریافت Context کامل ورزشکار برای سرویس FastAPI و آنتولوژی
    """
    def get(self, request, user_id):
        user = get_object_or_404(User, id=user_id)
        serializer = AthleteContextSerializer(user)
        return Response(serializer.data, status=status.HTTP_200_OK)


class UserProfileView(APIView):
    """
    اندپوینت دریافت و ویرایش مستقیم پروفایل جامع بیومکانیکی کاربر بر اساس user_id
    """
    def get(self, request, user_id):
        profile = get_object_or_404(UserProfile, user_id=user_id)
        serializer = UserProfileSerializer(profile)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def put(self, request, user_id):
        profile = get_object_or_404(UserProfile, user_id=user_id)
        serializer = UserProfileSerializer(profile, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def patch(self, request, user_id):
        profile = get_object_or_404(UserProfile, user_id=user_id)
        serializer = UserProfileSerializer(profile, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


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