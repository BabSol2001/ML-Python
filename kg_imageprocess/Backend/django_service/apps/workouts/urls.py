from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    ExerciseViewSet,
    WorkoutPlanViewSet,
    WorkoutPlanItemViewSet,
    CompletedSessionSummaryViewSet,
)

# تعریف روتر استاندارد REST API برای ماژول تمرینات
router = DefaultRouter()
router.register(r'exercises', ExerciseViewSet, basename='exercise')
router.register(r'plans', WorkoutPlanViewSet, basename='workout-plan')
router.register(r'plan-items', WorkoutPlanItemViewSet, basename='workout-plan-item')
router.register(r'summaries', CompletedSessionSummaryViewSet, basename='session-summary')

urlpatterns = [
    # شامل شدن تمام مسیرهای CRUD خودکار ماژول workouts
    path('', include(router.urls)),
]