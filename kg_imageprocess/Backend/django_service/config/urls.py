from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/users/', include('apps.users.urls')),
    path('api/workouts/', include('apps.workouts.urls')),
    path('api/payments/', include('apps.payments.urls')),
]