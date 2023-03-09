import dill
import glob
import pandas as pd
import json
import os
from datetime import datetime


# Предсказание для одного файла json
def prediction(json_test):
    data = pd.DataFrame([json_test])

    # Загрузка обученной модели
    with open(glob.glob("../data/models/*.pkl")[0], 'rb') as file:
        model = dill.load(file)

    y = model.predict(data)
    x = {'car_id': data.id, 'pred': y}
    df = pd.DataFrame(x)
    return df


# Предсказание для всех json-объектов из папки data/test
def predict():
    predicted_df = pd.DataFrame(columns=['car_id', 'pred'])
    for jsonfile in list(sorted(os.listdir('../data/test'))):
        with open(os.path.join('../data/test', jsonfile), 'r') as j:
            data = json.load(j)
            predicted_df = pd.concat([predicted_df, prediction(data)], axis=0)

    # Сохранение датафрейма предсказаний в csv файл в папку data/predictions
    predict_filename = f'../data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    predicted_df.to_csv(predict_filename, index=False)


if __name__ == '__main__':
    predict()
