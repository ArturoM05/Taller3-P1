from django.db import models
import numpy as np

def get_default_array():
    default_arr = np.random.rand(1536)
    return default_arr.tobytes()

class Movie(models.Model):
    title = models.CharField(max_length=100)
    description = models.CharField(max_length=250, null=True)
    image = models.ImageField(upload_to='movie/images/')
    url = models.URLField(blank=True)
    genre = models.CharField(max_length=250, blank=True)
    year = models.IntegerField(null=True, blank=True)
    emb = models.BinaryField(default=get_default_array())

    def __str__(self):
        return self.title