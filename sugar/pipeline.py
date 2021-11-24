"""
Interface to Speaker Recognition.

Author: Wang Rui
Version: v0.1
Create Date: 2020/9/27
"""


class SpeakerRecognition:

    def __init__(self):
        print("Start Machine Learning Pipeline for Speaker Recognition")
        print("Methods: ", end="")
        for m in self.__dir__():
            if "__" not in m:
                print(m, end=", ")
        print("")

    def DataPipeline(self):
        """
        Define data pipeline to feed model.
        """
        pass

    def Model(self):
        """
        Define a model.
        """
        pass

    def Train(self):
        """
        Define a training process.
        """
        pass

    def Report(self):
        """
        Define a reporter for evaluting the performace of a trained model.
        """
        pass

if __name__ == "__main__":
    sr = SpeakerRecognition()
