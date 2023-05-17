from django.conf import settings
from django.conf.urls.static import static

from . import views
from django.urls import path, include

urlpatterns = [
                  path('', views.index, name='index'),
                  path('download/', views.download_df, name='download_df'),
              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
