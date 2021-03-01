class Metrics:
    @staticmethod
    def accuracy_score(y_true, y_predict):
        counter = 0
        correct = 0
        for t, p in zip(y_true, y_predict):
            if t == p:
                correct += 1
            counter += 1
        return correct / counter