from django.urls import path
from . import views
from django.urls import path 
from django.conf import settings 
from django.conf.urls.static import static 

urlpatterns = [
        path('', views.index, name = 'index'),
	path('image_upload', views.index, name = 'image_upload'), 
]

if settings.DEBUG:
        urlpatterns += static(settings.MEDIA_URL,
                              document_root=settings.MEDIA_ROOT)
