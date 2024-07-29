import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


class IrisModelTrainer:

    def __init__(self):
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        iris = load_iris()
        X = iris.data
        y = iris.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

    def train_model(self):
        self.model = LogisticRegression(random_state=42, solver='liblinear')
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        f1 = f1_score(self.y_test, y_pred, average='macro')
        return f1

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)


if __name__ == "__main__":
    trainer = IrisModelTrainer()
    trainer.load_data()
    trainer.train_model()
    score = trainer.evaluate_model()
    print("F1 score:", score)
    trainer.save('iris_model.pkl')