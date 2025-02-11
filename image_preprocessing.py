from PIL import Image


class ImageResizer:
    def __init__(self):
        """
        Initialize the ImageResizer class with the path of the image.
        """
        pass

    def resize(self, width, height, image_path):
        """
        Resize the image to the specified width and height.
        
        Parameters:
        - width (int): The target width of the image.
        - height (int): The target height of the image.
        - keep_aspect_ratio (bool): If True, resizes while maintaining the aspect ratio.

        Returns:
        - Resized Image object.
        """
        self.image = Image.open(image_path)
        self.image = self.image.resize((width, height ))
        
        return self.image

    def save(self, output_path):
        """
        Save the resized image to the specified output path.
        """
        self.image.save(output_path)

# Example Usa
