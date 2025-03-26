import os
from django.core.management.base import BaseCommand
from movie.models import Movie

class Command(BaseCommand):
    help = "Update movie images in the database from the media/images folder"
    
    def handle(self, *args, **kwargs):
        # 📥 Ruta de la carpeta con las imágenes actualizadas
        images_folder = 'media/movie/images'  # ← Puedes cambiar el nombre si es necesario

        # ✅ Verifica si la carpeta existe
        if not os.path.exists(images_folder):
            self.stderr.write(f"Images folder '{images_folder}' not found.")
            return

        updated_count = 0

        # 📖 Recorremos todos los archivos en la carpeta de imágenes
        for filename in os.listdir(images_folder):
            if filename.startswith('m_') and filename.endswith('.png'):
                title = filename[2:-4]  # Extraemos el título de la película del nombre del archivo

                try:
                    # ❗ Aquí debes completar el código para buscar la película por título
                    movie = Movie.objects.get(title=title)
                    # ❗ Aquí debes actualizar la imagen de la película
                    movie.image = os.path.join(images_folder, filename)
                    movie.save()
                    updated_count += 1

                    self.stdout.write(self.style.SUCCESS(f"Updated image for: {title}"))

                except Movie.DoesNotExist:
                    self.stderr.write(f"Movie not found: {title}")
                except Exception as e:
                    self.stderr.write(f"Failed to update image for {title}: {str(e)}")

        # ✅ Al finalizar, muestra cuántas imágenes se actualizaron
        self.stdout.write(self.style.SUCCESS(f"Finished updating {updated_count} movie images from folder."))