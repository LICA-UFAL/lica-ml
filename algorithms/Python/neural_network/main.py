from LICA_NN import NeuralNetwork
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
        
    df = pd.read_csv('admission_predict.csv')

    df['GRE Score'] = (df['GRE Score'] - df['GRE Score'].min()) / (df['GRE Score'].max() - df['GRE Score'].min())
    df['TOEFL Score'] = (df['TOEFL Score'] - df['TOEFL Score'].min()) / (df['TOEFL Score'].max() - df['TOEFL Score'].min())
    df['University Rating'] = (df['University Rating'] - df['University Rating'].min()) / (df['University Rating'].max() - df['University Rating'].min())
    df['SOP'] = (df['SOP'] - df['SOP'].min()) / (df['SOP'].max() - df['SOP'].min())
    df['LOR '] = (df['LOR '] - df['LOR '].min()) / (df['LOR '].max() - df['LOR '].min())
    df['CGPA'] = (df['CGPA'] - df['CGPA'].min()) / (df['CGPA'].max() - df['CGPA'].min())
    df['Research'] = (df['Research'] - df['Research'].min()) / (df['Research'].max() - df['Research'].min())
    df['Chance of Admit '] = (df['Chance of Admit '] - df['Chance of Admit '].min()) / (df['Chance of Admit '].min() - df['Chance of Admit '].max())

    total_mean = df['Chance of Admit '].mean()

    df[(df[['Chance of Admit ']] >= total_mean)] = 1
    df[(df[['Chance of Admit ']] < total_mean)] = 0

    X = df.iloc[:,1:6].values
    Y = df['Chance of Admit '].values

    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)

    model = NeuralNetwork(5, 5, 1)

    epochs = 100

    print("Train in " + str(len(x_train)) + " samples, validating in " + str(len(x_test)) + " samples \n")

    for epoch in range(epochs):
        print("Epoch: " + str(epoch + 1) + "/" + str(epochs) + "\n")
        for index in range(len(x_train)):
            model.fit(x_train[index], y_train[index])

    error = 0
    hit = 0

    for i in range(len(x_test)):
        if (model.predict(x_test[i]) > 0.5):
            if (y_test[i] == 1.0):
                hit += 1
            else:
                error += 1
        else:
            if (y_test[i] == 0.0):
                hit += 1
            else:
                error += 1

    print("Accuracy: %.2f" % ((hit*100) / len(x_test)),"%")
