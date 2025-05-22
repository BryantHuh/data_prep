class Colors:
    BLACK = (0, 0, 0)
    GRAY = (128, 128, 128)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    PURPLE = (128, 0, 128)
    ORANGE = (255, 165, 0)
    CYAN = (0, 255, 255)
    BROWN = (165, 42, 42)

    @classmethod
    def get_colors(cls):
        return [
            cls.GRAY, cls.BLACK, cls.RED, cls.GREEN,
            cls.BLUE, cls.YELLOW, cls.PURPLE, cls.ORANGE,
            cls.CYAN, cls.BROWN, cls.WHITE
        ]