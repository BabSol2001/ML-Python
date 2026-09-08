from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    InternalAthleteContextView,
    UserProfileView,
    BiomechanicalSessionViewSet,
    FrameAnalysisViewSet
)

# ساخت روتر برای اندپوینت‌های جلسات و فریم‌های بیومکانیک
router = DefaultRouter()
router.register(r'sessions', BiomechanicalSessionViewSet, basename='session')
router.register(r'frames', FrameAnalysisViewSet, basename='frame')

urlpatterns = [
    # اندپوینت دریافت context جامع ورزشکار بر اساس ID برای سرویس FastAPI
    path('internal/athlete-context/<uuid:user_id>/', InternalAthleteContextView.as_view(), name='internal_athlete_context'),
    
    # اندپوینت دریافت و ویرایش مستقیم پروفایل کاربر
    path('profile/<uuid:user_id>/', UserProfileView.as_view(), name='user_profile'),
    
    # اندپوینت‌های مربوط به sessions/ و frames/
    path('', include(router.urls)),
]