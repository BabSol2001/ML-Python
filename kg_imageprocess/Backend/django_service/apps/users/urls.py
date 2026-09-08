from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    InternalAthleteContextView,
    BiomechanicalSessionViewSet,
    FrameAnalysisViewSet
)

# ساخت روتر برای اندپوینت‌های جلسات و فریم‌های بیومکانیک
router = DefaultRouter()
router.register(r'sessions', BiomechanicalSessionViewSet, basename='session')
router.register(r'frames', FrameAnalysisViewSet, basename='frame')

urlpatterns = [
    # اندپوینت دریافت context ورزشکار بر اساس ID
    path('internal/athlete-context/<uuid:user_id>/', InternalAthleteContextView.as_view(), name='internal_athlete_context'),
    
    # اندپوینت‌های مربوط به sessions/ و frames/
    path('', include(router.urls)),
]