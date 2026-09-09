from typing import Any
from django.db.models import QuerySet
from rest_framework import viewsets
from rest_framework.request import Request

from .models import Exercise, WorkoutPlan, WorkoutPlanItem, CompletedSessionSummary
from .serializers import (
    ExerciseSerializer,
    WorkoutPlanSerializer,
    WorkoutPlanItemSerializer,
    CompletedSessionSummarySerializer,
)


class ExerciseViewSet(viewsets.ModelViewSet):
    """
    مدیریت بانک مرجع حرکات ورزشی و مشخصات بیومکانیکی
    """
    queryset = Exercise.objects.all()
    serializer_class = ExerciseSerializer
    lookup_field = 'id'


class WorkoutPlanViewSet(viewsets.ModelViewSet):
    """
    ایجاد، مشاهده و مدیریت برنامه‌های تمرینی ورزشکاران
    """
    queryset = WorkoutPlan.objects.all()
    serializer_class = WorkoutPlanSerializer

    def get_queryset(self) -> Any:
        """امکان فیلتر برنامه‌ها بر اساس کاربر از طریق Query Parameter: ?user_id=..."""
        queryset = WorkoutPlan.objects.all()
        request: Request = self.request  # type: ignore[assignment]
        user_id = request.query_params.get('user_id')
        if user_id:
            queryset = queryset.filter(user_id=user_id)
        return queryset


class WorkoutPlanItemViewSet(viewsets.ModelViewSet):
    """
    مدیریت آیتم‌ها و حرکات داخل یک برنامه تمرینی
    """
    queryset = WorkoutPlanItem.objects.all()
    serializer_class = WorkoutPlanItemSerializer

    def get_queryset(self) -> Any:
        """فیلتر حرکات بر اساس برنامه تمرینی: ?plan_id=..."""
        queryset = WorkoutPlanItem.objects.all()
        request: Request = self.request  # type: ignore[assignment]
        plan_id = request.query_params.get('plan_id')
        if plan_id:
            queryset = queryset.filter(plan_id=plan_id)
        return queryset


class CompletedSessionSummaryViewSet(viewsets.ModelViewSet):
    """
    ثبت خلاصه عملکرد بیومکانیکی (توسط سرویس FastAPI) و دریافت تاریخچه تمرینات توسط کاربر
    """
    queryset = CompletedSessionSummary.objects.all()
    serializer_class = CompletedSessionSummarySerializer

    def get_queryset(self) -> Any:
        """فیلتر خلاصه جلسات بر اساس کاربر: ?user_id=..."""
        queryset = CompletedSessionSummary.objects.all()
        request: Request = self.request  # type: ignore[assignment]
        user_id = request.query_params.get('user_id')
        if user_id:
            queryset = queryset.filter(user_id=user_id)
        return queryset