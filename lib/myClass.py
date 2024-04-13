import numpy as np


class MySize:
    """矩形のサイズ"""

    def __init__(self, width, height):
        """コンストラクタ

        Args:
            width: 幅
            height: 高さ
        """
        self.width = width
        self.height = height


class MyRect:
    """矩形"""

    def __init__(self, x: int, y: int, width: int, height: int):
        """コンストラクタ"""
        self.x: int = x
        self.y: int = y
        self.width: int = width
        self.height: int = height

    def tl(self) -> tuple:
        """左上座標を返す"""
        return (self.x, self.y)

    def br(self) -> tuple:
        """右下座標を返す"""
        return (self.x + self.width, self.y + self.height)

    def size(self) -> tuple:
        """サイズを返す"""
        return MySize(self.width, self.height)

    def area(self) -> int:
        """面積を返す"""
        return self.width * self.height

    def show(self):
        """矩形の情報を出力する"""
        print("rect(x, y, width, height) = ({}, {}, {}, {})".format(
            self.x, self.y, self.width, self.height))


class Parameter:
    """ランダムマスキングのパラメータ"""

    def __init__(self, p: float, s_l: float, s_h: float, r_1: float, r_2: float):
        """コンストラクタ"""
        self.p = p
        self.s_l = s_l
        self.s_h = s_h
        self.r_1 = r_1
        self.r_2 = r_2


class RandomErase:
    """ランダムマスキング領域算出を扱うクラス"""

    def __init__(self, parameter: Parameter):
        """コンストラクタ

        Args:
            Parameter: ランダムマスキングのパラメータ
        """
        self.parameter = parameter

    def erase(self, rect: MyRect) -> MyRect:
        """ランダムマスキング領域を算出する

        Args:
            rect: ラベル付けされた矩形

        Return:
            ランダムマスキング領域
        """
        while True:
            s_e = np.random.uniform(
                self.parameter.s_l, self.parameter.s_h) * rect.area()
            r_e = np.random.uniform(self.parameter.r_1, self.parameter.r_2)
            h_e = np.sqrt(s_e * r_e)
            w_e = np.sqrt(s_e / r_e)
            x_e = np.random.uniform(0, rect.width)
            y_e = np.random.uniform(0, rect.height)
            if (x_e + w_e <= rect.width) and (y_e + h_e <= rect.height):
                return MyRect(int(x_e + rect.x), int(y_e + rect.y),
                              int(w_e), int(h_e))
