from django.conf import settings
from django.conf.urls.static import static

from . import views
from django.urls import path, include

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_file, name='upload_file'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


