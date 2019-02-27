import cv2


class SimplePreprocessor:
    # Method: Constructor
    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        """
        :param width: Chieu rong hinh anh
        :param height: Chieu cao anh
        :param interpolation: Thuat toan noi suy
        """
        self.width = width
        self.height = height
        self.interpolation = interpolation

    # Method: Được sử dụng để thay đổi kích thước hình ảnh thành kích thước cố định (bỏ qua tỷ lệ khung hình)
    def preprocess(self, image):
        """
        :param image: Image
        :return: Re-sized image
        """
        return cv2.resize(image, (self.width, self.height), interpolation=self.interpolation)