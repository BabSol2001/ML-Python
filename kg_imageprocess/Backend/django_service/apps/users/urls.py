from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    UserViewSet,
    UserProfileViewSet,
    UserProfileView,
    InternalAthleteContextView,
    BiomechanicalSessionViewSet,
    FrameAnalysisViewSet
)

# ساخت روتر استاندارد REST API برای ViewSetها
router = DefaultRouter()
router.register(r'accounts', UserViewSet, basename='user')
router.register(r'profiles', UserProfileViewSet, basename='profile')
router.register(r'sessions', BiomechanicalSessionViewSet, basename='session')
router.register(r'frames', FrameAnalysisViewSet, basename='frame')

urlpatterns = [
    # اندپوینت دریافت context جامع ورزشکار بر اساس ID برای سرویس FastAPI
    path('internal/athlete-context/<uuid:user_id>/', InternalAthleteContextView.as_view(), name='internal_athlete_context'),
    
    # اندپوینت دریافت و ویرایش مستقیم پروفایل کاربر بر اساس user_id
    path('profile/<uuid:user_id>/', UserProfileView.as_view(), name='user_profile'),
    
    # شامل شدن تمام مسیرهای CRUD خودکار روتر (کاربران، پروفایل‌ها، جلسات، فریم‌ها)
    path('', include(router.urls)),
]