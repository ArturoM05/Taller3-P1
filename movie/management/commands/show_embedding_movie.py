import os
import random
import numpy as np
from django.core.management.base import BaseCommand

from movie.models import Movie

class Command(BaseCommand):
    help = "muestra el embedding de una pelicula aleatoria"

    def handle(self, *args, **kwargs):
        movies = Movie.objects.exclude(emb=None)

        movie = random.choice(movies)
        
        print(movie.title)

        print(np.frombuffer(movie.emb, dtype=np.float32))
