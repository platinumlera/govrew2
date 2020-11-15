```python
import pandas as pd
import numpy as np


INPUT_FILE = 'C:/Users/SOSI/Downloads/dataset_clear.csv' 
TEXT_FIELD = 0 

# in_data = csv.reader(open(INPUT_FILE, encoding='utf8', newline=''), delimiter=';') 

in_data = pd.read_csv(INPUT_FILE, delimiter=';',decimal='.', encoding='utf8')

# this gives us the size of the array
print('Dataset size', in_data.shape)

# here we can get size of array as two variables{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "FileNotFoundError",
     "evalue": "[Errno 2] File b'/Users/polinailenkova/Downloads/dataset_clear.csv' does not exist: b'/Users/polinailenkova/Downloads/dataset_clear.csv'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-1-274d983466b2>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      8\u001b[0m \u001b[1;31m# in_data = csv.reader(open(INPUT_FILE, encoding='utf8', newline=''), delimiter=';')\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      9\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 10\u001b[1;33m \u001b[0min_data\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mread_csv\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mINPUT_FILE\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mdelimiter\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m';'\u001b[0m\u001b[1;33m,\u001b[0m\u001b[0mdecimal\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'.'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mencoding\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m'utf8'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     11\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     12\u001b[0m \u001b[1;31m# this gives us the size of the array\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\io\\parsers.py\u001b[0m in \u001b[0;36mparser_f\u001b[1;34m(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, dialect, error_bad_lines, warn_bad_lines, delim_whitespace, low_memory, memory_map, float_precision)\u001b[0m\n\u001b[0;32m    683\u001b[0m         )\n\u001b[0;32m    684\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 685\u001b[1;33m         \u001b[1;32mreturn\u001b[0m \u001b[0m_read\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfilepath_or_buffer\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mkwds\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    686\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    687\u001b[0m     \u001b[0mparser_f\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m__name__\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mname\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\io\\parsers.py\u001b[0m in \u001b[0;36m_read\u001b[1;34m(filepath_or_buffer, kwds)\u001b[0m\n\u001b[0;32m    455\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    456\u001b[0m     \u001b[1;31m# Create the parser.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 457\u001b[1;33m     \u001b[0mparser\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mTextFileReader\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfp_or_buf\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    458\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    459\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[0mchunksize\u001b[0m \u001b[1;32mor\u001b[0m \u001b[0miterator\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\io\\parsers.py\u001b[0m in \u001b[0;36m__init__\u001b[1;34m(self, f, engine, **kwds)\u001b[0m\n\u001b[0;32m    893\u001b[0m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0moptions\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m\"has_index_names\"\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mkwds\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m\"has_index_names\"\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    894\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 895\u001b[1;33m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_make_engine\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mengine\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    896\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    897\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0mclose\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\io\\parsers.py\u001b[0m in \u001b[0;36m_make_engine\u001b[1;34m(self, engine)\u001b[0m\n\u001b[0;32m   1133\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m_make_engine\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mengine\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;34m\"c\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1134\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[0mengine\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;34m\"c\"\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1135\u001b[1;33m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_engine\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mCParserWrapper\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mf\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0moptions\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1136\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1137\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mengine\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;34m\"python\"\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pandas\\io\\parsers.py\u001b[0m in \u001b[0;36m__init__\u001b[1;34m(self, src, **kwds)\u001b[0m\n\u001b[0;32m   1915\u001b[0m         \u001b[0mkwds\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m\"usecols\"\u001b[0m\u001b[1;33m]\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0musecols\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1916\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1917\u001b[1;33m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_reader\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mparsers\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mTextReader\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0msrc\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;33m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1918\u001b[0m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0munnamed_cols\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_reader\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0munnamed_cols\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1919\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\parsers.pyx\u001b[0m in \u001b[0;36mpandas._libs.parsers.TextReader.__cinit__\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;32mpandas\\_libs\\parsers.pyx\u001b[0m in \u001b[0;36mpandas._libs.parsers.TextReader._setup_parser_source\u001b[1;34m()\u001b[0m\n",
      "\u001b[1;31mFileNotFoundError\u001b[0m: [Errno 2] File b'/Users/polinailenkova/Downloads/dataset_clear.csv' does not exist: b'/Users/polinailenkova/Downloads/dataset_clear.csv'"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "\n",
    "INPUT_FILE = '/Users/polinailenkova/Downloads/dataset_clear.csv' \n",
    "TEXT_FIELD = 0 \n",
    "\n",
    "# in_data = csv.reader(open(INPUT_FILE, encoding='utf8', newline=''), delimiter=';') \n",
    "\n",
    "in_data = pd.read_csv(INPUT_FILE, delimiter=';',decimal='.', encoding='utf8')\n",
    "\n",
    "# this gives us the size of the array\n",
    "print('Dataset size', in_data.shape)\n",
    "\n",
    "# here we can get size of array as two variables\n",
    "num_rows, num_feature = in_data.shape\n",
    "\n",
    "print('row number: ', num_rows)\n",
    "print('feature number: ', num_feature)\n",
    "print('names of features: ', list(in_data))\n",
    "\n",
    "# ваши оценки лежат в колонке 'Eval', поэтому можно просто присвоить колонку вектору\n",
    "# Создаем вектор ответов\n",
    "Y = in_data['Eval'].values\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import csv \n",
    "import pymystem3 \n",
    "import regex\n",
    "\n",
    "empty = 0 \n",
    "\n",
    "# тут будет хранится список оригинальных текстов \n",
    "x_data = [] \n",
    "# тут будет хранится список лематизированных текстов \n",
    "y_data = [] \n",
    "\n",
    "\n",
    "# создание объекта mystem \n",
    "m = pymystem3.Mystem() \n",
    "# для простоты укажу 100 документов, если нужно полный набор отлематизировать, \n",
    "# то нужно поставить число документов, ну или num_rows\n",
    "for i in range(num_rows):\n",
    "    text = str(in_data['Message'][i]) \n",
    "    s = text.replace('\"', '') \n",
    "    s = m.lemmatize(s) \n",
    "    s = ''.join(s) \n",
    "    lemans=str(s) \n",
    "    lemans = regex.sub('[!@#$1234567890#]', '', lemans) \n",
    "\n",
    "    #print(''.join(lemmas))\n",
    "    print(i)\n",
    "    x_data.append(text) \n",
    "    y_data.append(lemans) \n",
    " \n",
    "print('Загрузка данных и процедура лематизации завершена')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# теперь проведем векторизацию\n",
    "# проводим векторизацию и удаление стоп слов\n",
    "# we define the file name with a list of stop words\n",
    "file_name ='/Users/polinailenkova/Downloads/my_stop.txt'\n",
    "# Please note that we upload a list of stop words in the data frame\n",
    "words = pd.read_csv(file_name, encoding='utf8')\n",
    "# now wa can convert dataframe into list\n",
    "#print('names of features: ', list(words))\n",
    "\n",
    "my_stop = words['Stop_words'].values.tolist()\n",
    "# we define list of word\n",
    "# print('the list of stop words: ', my_stop)\n",
    "\n",
    "#In this code, we create a procedure for transforming documents into a matrix based on a list of stop-stop words\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "vec = TfidfVectorizer(ngram_range=(1,1),stop_words=(my_stop))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# запускаем процедуру векторизации\n",
    "x = vec.fit_transform(y_data)\n",
    "# transform doc to matrix\n",
    "data = vec.fit_transform(y_data).toarray()\n",
    "print('процесс векторизации закончен')\n",
    "\n",
    "num_docs, num_feature =x.shape\n",
    "print('doc number: ', num_docs, 'feature number: ', num_feature)\n",
    "\n",
    "# print(y_data)\n",
    "myfich = vec.get_feature_names()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.feature_selection import SelectKBest\n",
    "from sklearn.feature_selection import chi2\n",
    "import matplotlib.pyplot as plt\n",
    "# число фич\n",
    "k=1000\n",
    "my_test = SelectKBest(chi2, k)\n",
    "my_fit = my_test.fit(x, Y)\n",
    "\n",
    "# определяем количество значимых цифр\n",
    "np.set_printoptions(precision=3)\n",
    "\n",
    "print('Fit score for all data: ', my_fit.scores_)\n",
    "\n",
    "plt.plot(my_fit.scores_,'-bo')\n",
    "plt.show()\n",
    "\n",
    "\n",
    "X_new = my_fit.transform(x)\n",
    "new_list = my_fit.get_support(indices=True)\n",
    "print('Размер массива нового массива', X_new.shape)\n",
    "print()\n",
    "print('Список оптимальных фич: ', new_list)\n",
    "\n",
    "print(new_list)\n",
    "\n",
    "print()\n",
    "# список фич и scores\n",
    "for i in range(len(new_list)-1):\n",
    "    j = new_list[i]\n",
    "    print(myfich[j],' - ', my_fit.scores_[j])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# фичи лежат в data\n",
    "# ответы в YY\n",
    "# Проведем разделение текстов на две части\n",
    "# импортируем поцедуру деления датетса на тестовую и обучающую коллекцию.\n",
    "from sklearn.model_selection import train_test_split\n",
    "# Делим нашу клддекцию на две части: 1. часть для обучения 2. часть для проверки результатов обучения\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.33, random_state=42)\n",
    "print('Test collection size: ', X_test.shape)\n",
    "print('Training collection size: ', X_train.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Обучаем модели: "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix\n",
    "from sklearn import metrics\n",
    "## Import the Classifier.\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "# устанавливаем число соседей\n",
    "n_neighbors=1\n",
    "\n",
    "# тип взвешивания: ‘uniform’ : uniform weights, ‘distance’ : weight points by the inverse of their distance\n",
    "weights = 'uniform'\n",
    "\n",
    "# тип расстояния (p = 1 - manhattan_distance, p = 2 - euclidean_distance, )\n",
    "p = 2\n",
    "\n",
    "# автоматический подбор типа алгоритма\n",
    "algorithm = 'auto'\n",
    "\n",
    "#Создаем модель\n",
    "knn = KNeighborsClassifier(n_neighbors, weights, algorithm, p)\n",
    "\n",
    "## Обучаем модель на тренироваочных данных\n",
    "knn.fit(X_train, y_train)\n",
    "\n",
    "# оцениваем точность нашей модели на основе известных значений y_train\n",
    "predicted = knn.predict(X_test)\n",
    "\n",
    "# расчитываем основные метрики качества\n",
    "print(metrics.classification_report(y_test, predicted))\n",
    "\n",
    "print('confusion matrix')\n",
    "print(metrics.confusion_matrix(y_test, predicted))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.metrics import f1_score\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "neighbors=10\n",
    "weights = 'uniform'\n",
    "p = 2\n",
    "#организуем пустой массив scores\n",
    "cv_scores_train = []\n",
    "cv_scores_test = []\n",
    "\n",
    "\n",
    "# в цикле перебираем число соседей и расчитываем среднее значение F1\n",
    "for k in range(1, neighbors):\n",
    "    knn = KNeighborsClassifier(k, weights, algorithm, p)\n",
    "    knn.fit(X_train, y_train)\n",
    "    # knn.score(X_test, y_test)\n",
    "    # подсчет среднего значения F1 при разных разбиениях\n",
    "    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='f1_macro')\n",
    "    # расчет среднего значения scores по всем разбиениям\n",
    "    #print(np.mean(scores))\n",
    "    cv_scores_train.append(np.mean(scores))\n",
    "    \n",
    "    scores = cross_val_score(knn, X_test, y_test, cv=5, scoring='f1_macro')\n",
    "    # добавляем в список очереное среднее значение f1_macro\n",
    "    cv_scores_test.append(np.mean(scores))\n",
    "   \n",
    "    # вывод среднего значения F1 по всем классам\n",
    "    #print(f1_score(y_test, knn.predict(X_test), average='macro'))\n",
    "    # расчитываем F1 и добавляем его в список\n",
    "    #cv_scores_test.append(f1_score(y_test, knn.predict(X_test), average='macro'))\n",
    "\n",
    "# строим график F1\n",
    "plt.plot(cv_scores_train, 'ro', alpha=0.2)\n",
    "plt.plot(cv_scores_train, '-bo', linewidth=1)\n",
    "\n",
    "plt.plot(cv_scores_test, 'ro', alpha=0.2)\n",
    "plt.plot(cv_scores_test, 'r', alpha=0.2)\n",
    "plt.show()    \n",
    "print('расчет закончен')\n",
    "#1 сосед топ"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "n_neighbors=4\n",
    "\n",
    "# тип взвешивания: ‘uniform’ : uniform weights, ‘distance’ : weight points by the inverse of their distance\n",
    "weights = 'uniform'\n",
    "\n",
    "# тип расстояния (p = 1 - manhattan_distance, p = 2 - euclidean_distance, )\n",
    "p = 2\n",
    "\n",
    "# автоматический подбор типа алгоритма\n",
    "algorithm = 'auto'\n",
    "\n",
    "#Создаем модель\n",
    "knn = KNeighborsClassifier(n_neighbors, weights, algorithm, p)\n",
    "\n",
    "## Обучаем модель на тренироваочных данных\n",
    "knn.fit(X_train, y_train)\n",
    "\n",
    "# оцениваем точность нашей модели на основе известных значений y_train\n",
    "predicted = knn.predict(X_test)\n",
    "\n",
    "# расчитываем основные метрики качества\n",
    "print(metrics.classification_report(y_test, predicted))\n",
    "\n",
    "print('confusion matrix')\n",
    "print(metrics.confusion_matrix(y_test, predicted))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "KNN не оч"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# импортируем модель\n",
    "from sklearn import metrics\n",
    "from sklearn.svm import SVC\n",
    "\n",
    "\n",
    "\n",
    "# определяем модель классификатора\n",
    "model = SVC()\n",
    "# обучаем модель на тренировочном датасете\n",
    "model.fit(X_train, y_train)\n",
    "# выводим на экран параметры модели.\n",
    "print(model)\n",
    "\n",
    "# строим предсказание\n",
    "predicted = model.predict(X_test)\n",
    "\n",
    "# определяем качество модели\n",
    "print(metrics.classification_report(y_test, predicted))\n",
    "# строим confusion matrix\n",
    "print(metrics.confusion_matrix(y_test, predicted))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn import svm\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "\n",
    "param_grid = [\n",
    "  {'C': [1, 10], 'kernel': ['linear']},\n",
    "  {'C': [1, 10], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},\n",
    " ]\n",
    "\n",
    "svc = svm.SVC(gamma=\"scale\")\n",
    "clf = GridSearchCV(svc, param_grid, cv=5)\n",
    "clf.fit(X_train, y_train)\n",
    "# выводим наилучший результат\n",
    "clf.best_params_\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# определяем модель классификатора\n",
    "model = SVC(C=10, kernel='linear')\n",
    "# обучаем модель на тренировочном датасете\n",
    "model.fit(X_train, y_train)\n",
    "# выводим на экран параметры модели.\n",
    "print(model)\n",
    "\n",
    "# строим предсказание\n",
    "predicted = model.predict(X_test)\n",
    "\n",
    "# определяем качество модели\n",
    "print(metrics.classification_report(y_test, predicted))\n",
    "# строим confusion matrix\n",
    "print(metrics.confusion_matrix(y_test, predicted))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "SVC ниче так 53"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "clf = RandomForestClassifier(\n",
    "    n_estimators=50,\n",
    "    criterion='gini',\n",
    "    max_depth=5,\n",
    "    min_samples_split=2,\n",
    "    min_samples_leaf=1,\n",
    "    min_weight_fraction_leaf=0.0,\n",
    "    max_features='auto',\n",
    "    max_leaf_nodes=None,\n",
    "    min_impurity_decrease=0.0,\n",
    "    min_impurity_split=None,\n",
    "    bootstrap=True,\n",
    "    oob_score=False,\n",
    "    n_jobs=-1,\n",
    "    random_state=0,\n",
    "    verbose=0,\n",
    "    warm_start=False,\n",
    "    class_weight='balanced')\n",
    "\n",
    "\n",
    "clf.fit(X_train, y_train)\n",
    "# строим предсказание\n",
    "predicted = clf.predict(X_test)\n",
    "\n",
    "# определяем качество модели\n",
    "print(metrics.classification_report(y_test, predicted))\n",
    "# строим confusion matrix\n",
    "print(metrics.confusion_matrix(y_test, predicted))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "param_grid = { \n",
    "    'n_estimators': [200, 500],\n",
    "    'max_features': ['auto', 'sqrt', 'log2'],\n",
    "    'max_depth' : [4,5,6,7,8],\n",
    "    'criterion' :['gini', 'entropy']\n",
    "}\n",
    "\n",
    "clf = RandomForestClassifier(\n",
    "    n_estimators=50,\n",
    "    criterion='gini',\n",
    "    max_depth=5,\n",
    "    min_samples_split=2,\n",
    "    min_samples_leaf=1,\n",
    "    min_weight_fraction_leaf=0.0,\n",
    "    max_features='auto',\n",
    "    max_leaf_nodes=None,\n",
    "    min_impurity_decrease=0.0,\n",
    "    min_impurity_split=None,\n",
    "    bootstrap=True,\n",
    "    oob_score=False,\n",
    "    n_jobs=-1,\n",
    "    random_state=0,\n",
    "    verbose=0,\n",
    "    warm_start=False,\n",
    "    class_weight='balanced')\n",
    "\n",
    "clf = GridSearchCV(estimator= clf, param_grid=param_grid, cv= 5, scoring='f1_macro')\n",
    "clf.fit(X_train, y_train)\n",
    "# выводим наилучший результат\n",
    "print('best parameters: ', clf.best_params_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "clf = RandomForestClassifier(\n",
    "    n_estimators=500,\n",
    "    criterion='entropy',\n",
    "    max_depth=8,\n",
    "    min_samples_split=2,\n",
    "    min_samples_leaf=1,\n",
    "    min_weight_fraction_leaf=0.0,\n",
    "    max_features='auto',\n",
    "    max_leaf_nodes=None,\n",
    "    min_impurity_decrease=0.0,\n",
    "    min_impurity_split=None,\n",
    "    bootstrap=True,\n",
    "    oob_score=False,\n",
    "    n_jobs=-1,\n",
    "    random_state=0,\n",
    "    verbose=0,\n",
    "    warm_start=False,\n",
    "    class_weight='balanced')\n",
    "\n",
    "\n",
    "clf.fit(X_train, y_train)\n",
    "# строим предсказание\n",
    "predicted = clf.predict(X_test)\n",
    "\n",
    "# определяем качество модели\n",
    "print(metrics.classification_report(y_test, predicted))\n",
    "# строим confusion matrix\n",
    "print(metrics.confusion_matrix(y_test, predicted))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "НЕПЛОХО 52\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# и так, мы обучили классификатор KNN (для простоты не делал поиск оптимального числа соседей)\n",
    "#  теперь читаем новые данные, по которым нужно сделать предсказание\n",
    " #читаем новые данные \n",
    "f = '/Users/polinailenkova/Desktop/data_new.csv'\n",
    "df1=pd.read_csv(f,  sep=';', encoding='utf8')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# this gives us the size of the array\n",
    "print('Dataset size', df1.shape)\n",
    "\n",
    "# here we can get size of array as two variables\n",
    "num_rows1, num_feature1 = df1.shape\n",
    "\n",
    "print('row number: ', num_rows1)\n",
    "print('feature number: ', num_feature1)\n",
    "print('names of features: ', list(df1))\n",
    "\n",
    "print('-------------------')\n",
    "print('full data loaded')\n",
    "print('-------------------')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# для простоты возьмем 200 текстов\n",
    "# проводим лематизацию этого датасета\n",
    "y_data1 = []\n",
    "# how we arrange cycle for lematization of 200 documents\n",
    "for i in range(500):\n",
    "    # getting doc\n",
    "    text = str(df1['Text'][i])\n",
    "    # remove quotes from text \n",
    "    s = text.replace('\"', '')\n",
    "    # how we are doing lematization\n",
    "    s = m.lemmatize(s)\n",
    "    s = ''.join(s)\n",
    "    lemans=str(s)\n",
    "    lemans = regex.sub('[!@#$1234567890#—ツ►๑۩۞۩•*”˜˜”*°°*`]', '', lemans)\n",
    "    y_data1.append(lemans)    \n",
    "    print('doc number : ', i)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# теперь проводим векторизацию используя уже обученный векторизатор\n",
    "count_vector=vec.transform(y_data1)\n",
    "\n",
    "# transform doc to matrix\n",
    "data1 = vec.transform(y_data1).toarray()\n",
    "print('процесс векторизации закончен')\n",
    "print('размер новой матрицы: ', data1.shape)\n",
    "data_new = my_fit.transform(data1)\n",
    "print(data_new.shape)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# теперь можно сделать предсказание\n",
    "# вот теперь подставить полученную матрицу data1 в уже обученный классификатор\n",
    "y_predicted1 = model.predict(data_new)\n",
    "# предсказание сентимента новых текстов.\n",
    "print(y_predicted1)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

num_rows, num_feature = in_data.shape

print('row number: ', num_rows)
print('feature number: ', num_feature)
print('names of features: ', list(in_data))

# ваши оценки лежат в колонке 'Eval', поэтому можно просто присвоить колонку вектору
# Создаем вектор ответов
Y = in_data['Eval'].values


```

    Dataset size (565, 2)
    row number:  565
    feature number:  2
    names of features:  ['Message', 'Eval']
    


```python
import csv 
import pymystem3 
import re

empty = 0 

# тут будет хранится список оригинальных текстов 
x_data = [] 
# тут будет хранится список лематизированных текстов 
y_data = [] 


# создание объекта mystem 
m = pymystem3.Mystem() 
# для простоты укажу 100 документов, если нужно полный набор отлематизировать, 
# то нужно поставить число документов, ну или num_rows
for i in range(num_rows):
    text = str(in_data['Message'][i]) 
    s = text.replace('"', '') 
    s = m.lemmatize(s) 
    s = ''.join(s) 
    lemans=str(s) 
    lemans = re.sub('[!@#$1234567890#]', '', lemans) 

    #print(''.join(lemmas))
    print(i)
    x_data.append(text) 
    y_data.append(lemans) 
 
print('Загрузка данных и процедура лематизации завершена')
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    16
    17
    18
    19
    20
    21
    22
    23
    24
    25
    26
    27
    28
    29
    30
    31
    32
    33
    34
    35
    36
    37
    38
    39
    40
    41
    42
    43
    44
    45
    46
    47
    48
    49
    50
    51
    52
    53
    54
    55
    56
    57
    58
    59
    60
    61
    62
    63
    64
    65
    66
    67
    68
    69
    70
    71
    72
    73
    74
    75
    76
    77
    78
    79
    80
    81
    82
    83
    84
    85
    86
    87
    88
    89
    90
    91
    92
    93
    94
    95
    96
    97
    98
    99
    100
    101
    102
    103
    104
    105
    106
    107
    108
    109
    110
    111
    112
    113
    114
    115
    116
    117
    118
    119
    120
    121
    122
    123
    124
    125
    126
    127
    128
    129
    130
    131
    132
    133
    134
    135
    136
    137
    138
    139
    140
    141
    142
    143
    144
    145
    146
    147
    148
    149
    150
    151
    152
    153
    154
    155
    156
    157
    158
    159
    160
    161
    162
    163
    164
    165
    166
    167
    168
    169
    170
    171
    172
    173
    174
    175
    176
    177
    178
    179
    180
    181
    182
    183
    184
    185
    186
    187
    188
    189
    190
    191
    192
    193
    194
    195
    196
    197
    198
    199
    200
    201
    202
    203
    204
    205
    206
    207
    208
    209
    210
    211
    212
    213
    214
    215
    216
    217
    218
    219
    220
    221
    222
    223
    224
    225
    226
    227
    228
    229
    230
    231
    232
    233
    234
    235
    236
    237
    238
    239
    240
    241
    242
    243
    244
    245
    246
    247
    248
    249
    250
    251
    252
    253
    254
    255
    256
    257
    258
    259
    260
    261
    262
    263
    264
    265
    266
    267
    268
    269
    270
    271
    272
    273
    274
    275
    276
    277
    278
    279
    280
    281
    282
    283
    284
    285
    286
    287
    288
    289
    290
    291
    292
    293
    294
    295
    296
    297
    298
    299
    300
    301
    302
    303
    304
    305
    306
    307
    308
    309
    310
    311
    312
    313
    314
    315
    316
    317
    318
    319
    320
    321
    322
    323
    324
    325
    326
    327
    328
    329
    330
    331
    332
    333
    334
    335
    336
    337
    338
    339
    340
    341
    342
    343
    344
    345
    346
    347
    348
    349
    350
    351
    352
    353
    354
    355
    356
    357
    358
    359
    360
    361
    362
    363
    364
    365
    366
    367
    368
    369
    370
    371
    372
    373
    374
    375
    376
    377
    378
    379
    380
    381
    382
    383
    384
    385
    386
    387
    388
    389
    390
    391
    392
    393
    394
    395
    396
    397
    398
    399
    400
    401
    402
    403
    404
    405
    406
    407
    408
    409
    410
    411
    412
    413
    414
    415
    416
    417
    418
    419
    420
    421
    422
    423
    424
    425
    426
    427
    428
    429
    430
    431
    432
    433
    434
    435
    436
    437
    438
    439
    440
    441
    442
    443
    444
    445
    446
    447
    448
    449
    450
    451
    452
    453
    454
    455
    456
    457
    458
    459
    460
    461
    462
    463
    464
    465
    466
    467
    468
    469
    470
    471
    472
    473
    474
    475
    476
    477
    478
    479
    480
    481
    482
    483
    484
    485
    486
    487
    488
    489
    490
    491
    492
    493
    494
    495
    496
    497
    498
    499
    500
    501
    502
    503
    504
    505
    506
    507
    508
    509
    510
    511
    512
    513
    514
    515
    516
    517
    518
    519
    520
    521
    522
    523
    524
    525
    526
    527
    528
    529
    530
    531
    532
    533
    534
    535
    536
    537
    538
    539
    540
    541
    542
    543
    544
    545
    546
    547
    548
    549
    550
    551
    552
    553
    554
    555
    556
    557
    558
    559
    560
    561
    562
    563
    564
    Загрузка данных и процедура лематизации завершена
    


```python
display(in_data)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Message</th>
      <th>Eval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Огромное спасибо всем кто приложил к созданию ...</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Ни разу не смогла найти нужный приказ, даже ко...</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>Некоторые разделы сайта открываются совсем с д...</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>не нашел сведений о доходах гос. служащих Мин ...</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>Опубликуйте юбилейные даты 2013 года.</td>
      <td>1</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>560</td>
      <td>По услуге Водотведение в регионе Иркутская обл...</td>
      <td>0</td>
    </tr>
    <tr>
      <td>561</td>
      <td>Объясните, почему на сайте федеральной службы ...</td>
      <td>0</td>
    </tr>
    <tr>
      <td>562</td>
      <td>Не могу найти текущую ставку тарифа за подключ...</td>
      <td>0</td>
    </tr>
    <tr>
      <td>563</td>
      <td>Сайт устроен неудобно. Поисковик работает неад...</td>
      <td>0</td>
    </tr>
    <tr>
      <td>564</td>
      <td>Почему в калькуляторе коммунальных платежей (о...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>565 rows × 2 columns</p>
</div>



```python
# теперь проведем векторизацию
# проводим векторизацию и удаление стоп слов
# we define the file name with a list of stop words
file_name ='C:/Users/SOSI/Downloads/my_stop.txt'
# Please note that we upload a list of stop words in the data frame
words = pd.read_csv(file_name, encoding='utf8')
# now wa can convert dataframe into list
#print('names of features: ', list(words))

my_stop = words['Stop_words'].values.tolist()
# we define list of word
# print('the list of stop words: ', my_stop)

#In this code, we create a procedure for transforming documents into a matrix based on a list of stop-stop words
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(ngram_range=(1,1),stop_words=(my_stop))
```


```python
# запускаем процедуру векторизации
x = vec.fit_transform(y_data)
# transform doc to matrix
data = vec.fit_transform(y_data).toarray()
print('процесс векторизации закончен')

num_docs, num_feature =x.shape
print('doc number: ', num_docs, 'feature number: ', num_feature)

# print(y_data)
myfich = vec.get_feature_names()
```

    процесс векторизации закончен
    doc number:  565 feature number:  2729
    


```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
# число фич
k=1000
my_test = SelectKBest(chi2, k)
my_fit = my_test.fit(x, Y)

# определяем количество значимых цифр
np.set_printoptions(precision=3)

print('Fit score for all data: ', my_fit.scores_)

plt.plot(my_fit.scores_,'-bo')
plt.show()


X_new = my_fit.transform(x)
new_list = my_fit.get_support(indices=True)
print('Размер массива нового массива', X_new.shape)
print()
print('Список оптимальных фич: ', new_list)

print(new_list)

print()
# список фич и scores
for i in range(len(new_list)-1):
    j = new_list[i]
    print(myfich[j],' - ', my_fit.scores_[j])
```

    Fit score for all data:  [0.506 0.506 1.201 ... 0.237 0.218 1.25 ]
    


![png](output_5_1.png)


    Размер массива нового массива (565, 1000)
    
    Список оптимальных фич:  [   2    8    9   10   12   16   20   22   24   27   28   29   37   41
       44   47   59   62   66   74   77   80   82   84   91   92   98  102
      103  115  119  123  124  126  128  138  141  144  146  148  156  157
      170  171  173  174  176  179  180  183  187  192  193  197  198  200
      201  204  205  207  209  211  212  213  214  228  229  234  235  236
      238  239  241  242  249  251  252  255  262  263  264  265  270  275
      278  282  283  287  295  296  297  299  301  304  308  313  317  319
      320  322  332  340  347  349  357  359  360  364  367  369  372  378
      380  386  387  391  400  401  406  411  412  413  414  421  424  427
      432  433  439  441  442  451  452  456  457  458  459  460  464  472
      473  474  475  476  477  481  484  485  489  490  493  496  499  500
      507  515  516  520  524  525  527  530  531  533  536  537  538  539
      541  542  551  553  559  560  562  565  567  571  574  576  577  581
      588  589  591  593  600  602  603  607  608  614  619  623  624  625
      626  630  632  633  635  636  640  641  642  646  648  649  651  652
      654  655  657  660  662  665  666  675  678  679  686  687  690  695
      696  702  703  706  712  714  716  717  720  722  724  731  733  740
      744  748  750  754  757  758  765  766  767  769  770  772  774  775
      776  777  778  779  781  782  783  787  788  789  800  801  803  805
      808  809  811  813  816  820  822  828  829  830  832  834  836  840
      841  844  845  846  850  855  856  861  862  863  870  873  874  875
      879  880  888  890  892  893  895  897  900  903  904  905  910  913
      926  927  931  934  935  936  937  947  951  952  957  958  960  961
      967  968  971  973  974  975  978  979  980  986  988  990  991  998
     1001 1002 1005 1007 1010 1012 1013 1015 1016 1017 1019 1024 1025 1028
     1029 1031 1035 1037 1038 1043 1048 1049 1050 1060 1063 1064 1067 1068
     1069 1072 1074 1076 1077 1081 1086 1089 1090 1092 1095 1098 1101 1104
     1105 1106 1107 1113 1115 1116 1117 1119 1120 1123 1138 1142 1144 1145
     1151 1152 1155 1156 1159 1163 1168 1172 1174 1175 1178 1186 1188 1189
     1190 1192 1193 1198 1202 1204 1205 1208 1209 1210 1212 1213 1215 1217
     1221 1229 1230 1237 1238 1240 1241 1244 1246 1247 1250 1256 1258 1261
     1262 1264 1266 1268 1273 1276 1277 1279 1280 1282 1283 1288 1290 1299
     1306 1309 1313 1314 1315 1320 1321 1323 1326 1327 1330 1344 1348 1349
     1350 1352 1354 1356 1357 1359 1360 1363 1366 1368 1370 1373 1375 1385
     1389 1390 1392 1393 1394 1395 1402 1406 1408 1411 1412 1417 1418 1419
     1421 1423 1425 1430 1433 1434 1436 1438 1439 1440 1442 1449 1450 1451
     1453 1455 1460 1461 1463 1464 1467 1474 1475 1476 1477 1481 1488 1489
     1490 1491 1495 1497 1499 1502 1503 1510 1511 1512 1513 1516 1518 1519
     1520 1521 1522 1523 1524 1527 1532 1533 1537 1546 1550 1551 1554 1562
     1563 1566 1567 1569 1570 1571 1576 1577 1578 1581 1583 1586 1587 1588
     1589 1590 1594 1595 1596 1599 1601 1602 1603 1604 1607 1609 1612 1613
     1614 1616 1624 1625 1629 1633 1635 1638 1639 1641 1646 1649 1651 1654
     1655 1657 1658 1660 1662 1664 1665 1669 1671 1672 1678 1679 1682 1684
     1685 1687 1688 1689 1695 1699 1700 1703 1706 1714 1725 1726 1727 1732
     1734 1739 1741 1742 1745 1748 1750 1754 1755 1757 1759 1761 1763 1764
     1765 1769 1774 1776 1777 1778 1779 1780 1791 1793 1803 1806 1807 1813
     1816 1820 1827 1829 1830 1831 1833 1834 1835 1837 1841 1845 1846 1848
     1852 1853 1855 1856 1862 1864 1865 1866 1867 1868 1869 1873 1874 1878
     1882 1886 1887 1890 1891 1894 1896 1903 1915 1917 1919 1921 1924 1926
     1927 1930 1932 1933 1936 1940 1943 1945 1947 1953 1959 1961 1962 1966
     1969 1971 1972 1975 1976 1979 1981 1982 1989 1991 1996 2001 2002 2005
     2006 2009 2010 2011 2012 2015 2020 2022 2023 2026 2034 2036 2037 2039
     2040 2041 2042 2044 2045 2046 2048 2050 2051 2055 2056 2057 2061 2062
     2066 2068 2069 2071 2075 2081 2083 2084 2086 2087 2088 2090 2094 2096
     2097 2104 2109 2110 2112 2114 2115 2118 2124 2125 2126 2127 2133 2138
     2142 2146 2148 2149 2150 2154 2156 2158 2160 2161 2162 2164 2166 2175
     2176 2178 2180 2183 2185 2186 2187 2190 2194 2195 2198 2199 2202 2203
     2209 2211 2212 2214 2221 2224 2226 2231 2232 2234 2237 2238 2239 2241
     2242 2243 2244 2246 2257 2262 2269 2271 2274 2275 2276 2277 2280 2281
     2290 2292 2294 2295 2298 2299 2302 2303 2308 2309 2310 2311 2312 2313
     2314 2315 2317 2319 2320 2321 2325 2326 2337 2340 2344 2345 2346 2348
     2349 2354 2355 2357 2362 2366 2370 2371 2373 2375 2379 2381 2384 2386
     2397 2399 2404 2406 2410 2412 2413 2414 2415 2416 2417 2418 2419 2420
     2421 2423 2426 2427 2428 2429 2430 2439 2441 2442 2443 2444 2445 2447
     2449 2450 2451 2454 2457 2458 2459 2460 2461 2464 2465 2466 2469 2475
     2483 2484 2486 2494 2498 2499 2505 2506 2507 2509 2511 2512 2514 2516
     2522 2523 2526 2527 2530 2534 2536 2540 2544 2547 2548 2552 2553 2554
     2558 2565 2567 2568 2572 2575 2578 2583 2587 2592 2594 2599 2600 2604
     2605 2606 2610 2611 2613 2615 2618 2619 2621 2623 2627 2629 2633 2634
     2640 2644 2645 2654 2661 2664 2665 2667 2668 2669 2670 2674 2676 2677
     2678 2679 2680 2681 2683 2684 2694 2706 2707 2712 2713 2714 2716 2717
     2719 2721 2723 2724 2725 2728]
    [   2    8    9   10   12   16   20   22   24   27   28   29   37   41
       44   47   59   62   66   74   77   80   82   84   91   92   98  102
      103  115  119  123  124  126  128  138  141  144  146  148  156  157
      170  171  173  174  176  179  180  183  187  192  193  197  198  200
      201  204  205  207  209  211  212  213  214  228  229  234  235  236
      238  239  241  242  249  251  252  255  262  263  264  265  270  275
      278  282  283  287  295  296  297  299  301  304  308  313  317  319
      320  322  332  340  347  349  357  359  360  364  367  369  372  378
      380  386  387  391  400  401  406  411  412  413  414  421  424  427
      432  433  439  441  442  451  452  456  457  458  459  460  464  472
      473  474  475  476  477  481  484  485  489  490  493  496  499  500
      507  515  516  520  524  525  527  530  531  533  536  537  538  539
      541  542  551  553  559  560  562  565  567  571  574  576  577  581
      588  589  591  593  600  602  603  607  608  614  619  623  624  625
      626  630  632  633  635  636  640  641  642  646  648  649  651  652
      654  655  657  660  662  665  666  675  678  679  686  687  690  695
      696  702  703  706  712  714  716  717  720  722  724  731  733  740
      744  748  750  754  757  758  765  766  767  769  770  772  774  775
      776  777  778  779  781  782  783  787  788  789  800  801  803  805
      808  809  811  813  816  820  822  828  829  830  832  834  836  840
      841  844  845  846  850  855  856  861  862  863  870  873  874  875
      879  880  888  890  892  893  895  897  900  903  904  905  910  913
      926  927  931  934  935  936  937  947  951  952  957  958  960  961
      967  968  971  973  974  975  978  979  980  986  988  990  991  998
     1001 1002 1005 1007 1010 1012 1013 1015 1016 1017 1019 1024 1025 1028
     1029 1031 1035 1037 1038 1043 1048 1049 1050 1060 1063 1064 1067 1068
     1069 1072 1074 1076 1077 1081 1086 1089 1090 1092 1095 1098 1101 1104
     1105 1106 1107 1113 1115 1116 1117 1119 1120 1123 1138 1142 1144 1145
     1151 1152 1155 1156 1159 1163 1168 1172 1174 1175 1178 1186 1188 1189
     1190 1192 1193 1198 1202 1204 1205 1208 1209 1210 1212 1213 1215 1217
     1221 1229 1230 1237 1238 1240 1241 1244 1246 1247 1250 1256 1258 1261
     1262 1264 1266 1268 1273 1276 1277 1279 1280 1282 1283 1288 1290 1299
     1306 1309 1313 1314 1315 1320 1321 1323 1326 1327 1330 1344 1348 1349
     1350 1352 1354 1356 1357 1359 1360 1363 1366 1368 1370 1373 1375 1385
     1389 1390 1392 1393 1394 1395 1402 1406 1408 1411 1412 1417 1418 1419
     1421 1423 1425 1430 1433 1434 1436 1438 1439 1440 1442 1449 1450 1451
     1453 1455 1460 1461 1463 1464 1467 1474 1475 1476 1477 1481 1488 1489
     1490 1491 1495 1497 1499 1502 1503 1510 1511 1512 1513 1516 1518 1519
     1520 1521 1522 1523 1524 1527 1532 1533 1537 1546 1550 1551 1554 1562
     1563 1566 1567 1569 1570 1571 1576 1577 1578 1581 1583 1586 1587 1588
     1589 1590 1594 1595 1596 1599 1601 1602 1603 1604 1607 1609 1612 1613
     1614 1616 1624 1625 1629 1633 1635 1638 1639 1641 1646 1649 1651 1654
     1655 1657 1658 1660 1662 1664 1665 1669 1671 1672 1678 1679 1682 1684
     1685 1687 1688 1689 1695 1699 1700 1703 1706 1714 1725 1726 1727 1732
     1734 1739 1741 1742 1745 1748 1750 1754 1755 1757 1759 1761 1763 1764
     1765 1769 1774 1776 1777 1778 1779 1780 1791 1793 1803 1806 1807 1813
     1816 1820 1827 1829 1830 1831 1833 1834 1835 1837 1841 1845 1846 1848
     1852 1853 1855 1856 1862 1864 1865 1866 1867 1868 1869 1873 1874 1878
     1882 1886 1887 1890 1891 1894 1896 1903 1915 1917 1919 1921 1924 1926
     1927 1930 1932 1933 1936 1940 1943 1945 1947 1953 1959 1961 1962 1966
     1969 1971 1972 1975 1976 1979 1981 1982 1989 1991 1996 2001 2002 2005
     2006 2009 2010 2011 2012 2015 2020 2022 2023 2026 2034 2036 2037 2039
     2040 2041 2042 2044 2045 2046 2048 2050 2051 2055 2056 2057 2061 2062
     2066 2068 2069 2071 2075 2081 2083 2084 2086 2087 2088 2090 2094 2096
     2097 2104 2109 2110 2112 2114 2115 2118 2124 2125 2126 2127 2133 2138
     2142 2146 2148 2149 2150 2154 2156 2158 2160 2161 2162 2164 2166 2175
     2176 2178 2180 2183 2185 2186 2187 2190 2194 2195 2198 2199 2202 2203
     2209 2211 2212 2214 2221 2224 2226 2231 2232 2234 2237 2238 2239 2241
     2242 2243 2244 2246 2257 2262 2269 2271 2274 2275 2276 2277 2280 2281
     2290 2292 2294 2295 2298 2299 2302 2303 2308 2309 2310 2311 2312 2313
     2314 2315 2317 2319 2320 2321 2325 2326 2337 2340 2344 2345 2346 2348
     2349 2354 2355 2357 2362 2366 2370 2371 2373 2375 2379 2381 2384 2386
     2397 2399 2404 2406 2410 2412 2413 2414 2415 2416 2417 2418 2419 2420
     2421 2423 2426 2427 2428 2429 2430 2439 2441 2442 2443 2444 2445 2447
     2449 2450 2451 2454 2457 2458 2459 2460 2461 2464 2465 2466 2469 2475
     2483 2484 2486 2494 2498 2499 2505 2506 2507 2509 2511 2512 2514 2516
     2522 2523 2526 2527 2530 2534 2536 2540 2544 2547 2548 2552 2553 2554
     2558 2565 2567 2568 2572 2575 2578 2583 2587 2592 2594 2599 2600 2604
     2605 2606 2610 2611 2613 2615 2618 2619 2621 2623 2627 2629 2633 2634
     2640 2644 2645 2654 2661 2664 2665 2667 2668 2669 2670 2674 2676 2677
     2678 2679 2680 2681 2683 2684 2694 2706 2707 2712 2713 2714 2716 2717
     2719 2721 2723 2724 2725 2728]
    
    _dedef  -  1.2014763014008831
    amomosreg  -  0.6958888263761897
    answers  -  1.6448040367783865
    article  -  1.3063857077493926
    catid  -  0.7662784451342981
    com_content  -  1.3063857077493926
    customs  -  1.9092230676494868
    dbde  -  2.4454316724174734
    depgosregulirineconomy  -  1.2782773545825903
    doc_  -  1.2782773545825903
    docs  -  1.4980263435779813
    document  -  0.7155484663895009
    english  -  0.9367482091695336
    err_connection_timed_out  -  1.150494487697588
    excel  -  0.9896677783893539
    eximport  -  0.6565878183428813
    fssp  -  0.6267350949492793
    fstrf  -  1.6448040367783865
    gosuslugi  -  0.721176478755788
    id  -  1.3063857077493926
    index  -  0.9084488769658478
    interdocs  -  1.4980263435779813
    is  -  1.1145970397081377
    itemid  -  1.5325568902685962
    lkpr  -  0.999694196392073
    madr_sys  -  1.4980263435779813
    minec  -  1.2782773545825903
    moscow  -  0.999694196392073
    ms  -  0.5922556089308869
    option  -  1.3063857077493926
    passportpermit  -  0.6418403128716426
    phonebook  -  1.1145970397081377
    php  -  1.0438197676993313
    popup  -  0.6565878183428813
    qiwi  -  2.8101899859101516
    regions  -  1.6448040367783865
    rosminzdrav  -  1.150494487697588
    rss  -  1.4532297092699753
    ru  -  0.6032231604788997
    rupto  -  1.4980263435779813
    structure  -  1.2782773545825903
    svedenija_o_zaderzhkah  -  1.0012660780448517
    ved_info  -  0.6565878183428813
    version  -  0.9367482091695336
    view  -  1.3063857077493926
    vin  -  1.0326261242896069
    vozdushnye_perevozki  -  1.0012660780448517
    withoutvisa  -  0.6179826961949022
    wltgfy  -  1.0326261242896069
    xbkdcwzogsako  -  1.2191940541559303
    zafira  -  1.0326261242896069
    август  -  0.885383117625725
    автивация  -  0.6614062840644256
    автомобиль  -  1.0326261242896069
    автомобильopel  -  1.0326261242896069
    авторизация  -  0.6148225304499055
    авторизоваться  -  0.997257834803109
    административный  -  1.3696609327559695
    администратор  -  2.4454316724174734
    адрес  -  0.8359684379597978
    ай  -  0.6348982699513213
    аккредитация  -  2.272096205475236
    аккредитовать  -  1.2797092990243
    активация  -  0.6153669465446766
    активизируеться  -  0.856654375942677
    ананьев  -  0.7853835636589812
    анапа  -  0.9289793708866114
    аннулировать  -  1.0326261242896069
    анонс  -  0.8558901313833194
    антинаркотический  -  1.2618872221694657
    аппликация  -  0.8203346076293703
    апрель  -  2.3820341293611036
    арест  -  0.6289250010763889
    арестовывать  -  0.7253832379284699
    ассортимент  -  1.7440985826554554
    аттестат  -  0.77798536618509
    аудио  -  0.687893362342421
    байка  -  0.792718039624636
    баннер  -  0.9595554316851612
    бардак  -  0.6580056328480555
    башкиров  -  1.576074817040058
    башкортостан  -  0.9794277886654287
    безрезультатно  -  0.7982554366152912
    беременый  -  0.8666898741079008
    бесполезный  -  1.293494159929829
    благодарить  -  0.891776260072756
    благодарность  -  1.058737532342921
    бланк  -  1.5963547965054832
    больше  -  0.6133708182546562
    большереченский  -  1.3427176840815558
    большеуковский  -  1.3427176840815558
    большой  -  1.3876670022489734
    бороться  -  0.6453501924872029
    брать  -  1.948840222710165
    бровченко  -  1.747364601086053
    будущий  -  0.834597713144549
    бухгалтерский  -  0.6216823874316262
    бывший  -  1.936258157672531
    быстро  -  2.6957362228867794
    бытовой  -  1.7440985826554554
    вариант  -  0.7785399122343215
    ввод  -  0.9138169167170269
    ведомство  -  1.1487668496905663
    ведь  -  0.8067423671870099
    версия  -  1.006979838437548
    верхний  -  1.8482596068863146
    весна  -  0.6013596942105434
    весьма  -  0.8660265166246273
    вещество  -  0.6088640298305359
    взиматься  -  0.9372524296015834
    взыскатель  -  0.6837660741366693
    видимому  -  1.260854731838191
    видный  -  2.66230492997297
    виртуальный  -  0.630305600765289
    вирус  -  0.7267062348526508
    вкладка  -  0.6160865356287717
    вместо  -  1.301286798588055
    внесение  -  0.7754448540959576
    вносить  -  1.646954916497625
    водоканал  -  1.7672290147876932
    водоотведение  -  0.9666481230013241
    водостабжение  -  0.9154129101641291
    водотведение  -  0.9154129101641291
    возиться  -  0.6742677758886515
    возможно  -  2.4809700873383553
    возмущать  -  0.5916106202985001
    вологодский  -  0.6742677758886515
    вообще  -  0.7985104070485423
    воспользоваться  -  3.145207638161403
    восстанавливать  -  0.6598970657631726
    восстановление  -  0.6035966621971177
    вплывать  -  0.6160865356287717
    вполне  -  0.8977349925302183
    временный  -  0.5997631649545637
    временый  -  0.8666898741079008
    время  -  1.0716620296982682
    вручную  -  0.6184184297972092
    все  -  0.8416725453881827
    всплывать  -  0.6160865356287717
    вх  -  1.057483651888801
    вход  -  0.654824824254439
    входить  -  2.02474304284887
    вчера  -  0.6289250010763889
    въезд  -  3.130226473016422
    въезжать  -  0.936538344059834
    выводиться  -  1.0879830484129611
    выгружаться  -  1.260854731838191
    выгрузка  -  1.2428431268280433
    выдворение  -  2.4563203011507713
    выделять  -  1.4510514752631025
    вызывать  -  1.0141168565492495
    вылезать  -  0.7786325549137884
    выпадать  -  1.8033384613664685
    выписка  -  2.0740602415338394
    выражать  -  1.058737532342921
    высчитывать  -  3.4091015859600176
    вытесняться  -  0.8558901313833194
    выявлять  -  1.9165737468159192
    газ  -  1.585436079249272
    газпром  -  1.057483651888801
    галина  -  1.3896775193932165
    гарантия  -  0.6715918517566881
    где  -  1.1809639898925084
    германия  -  0.9372524296015834
    главное  -  1.4780361973595721
    главный  -  0.9993253000566796
    глубоко  -  1.3576349687893812
    глупость  -  0.9964814341148005
    го  -  0.632432356779463
    говорить  -  0.7062149541592831
    голосовой  -  0.8288847293580721
    гораздо  -  0.657447675732528
    госпа  -  1.1584519420739245
    господин  -  0.8464546734901707
    госпрограмма  -  0.9183089253492959
    госуслуга  -  0.7010760720592555
    готовность  -  2.194235750703408
    гражданско  -  0.6179826961949022
    грамотность  -  1.0345008075719782
    граф  -  0.6179826961949022
    графа  -  1.0739770675272935
    грубый  -  1.042345127772322
    да  -  0.7956503316815176
    давать  -  1.0419338844214185
    давнешний  -  0.8953709813520178
    давный  -  0.8953709813520178
    данные  -  0.6691389337293063
    дата  -  1.449958786753079
    два  -  1.060377958687095
    двор  -  1.3134778239837557
    двухсторонний  -  1.4873180738398826
    действительностьа  -  0.6778201605161451
    декларация  -  0.7909145733844465
    дело  -  1.5514047675722487
    делопроизводство  -  0.8239867645075885
    ден  -  0.8551787593331528
    день  -  1.0688907127344935
    десятый  -  1.348535551777303
    деяние  -  0.6727077884750141
    деятельность  -  2.59602100162625
    дизайн  -  0.9292879883157865
    дико  -  0.7786325549137884
    дмитрий  -  1.5933387380324213
    добавка  -  0.7786325549137884
    добавление  -  0.8234240762309322
    добросовестный  -  0.8288847293580721
    добыча  -  1.057483651888801
    довольный  -  0.7490773542781122
    дождаться  -  0.8318899598423392
    дозваниваться  -  1.4347360492845649
    дозвниться  -  0.7924304593067655
    доказательство  -  0.6289250010763889
    документ  -  4.072639508247784
    дол  -  1.3071232662903687
    долгий  -  0.5905303634882597
    должник  -  1.3609846812707855
    должно  -  2.0535235870220605
    дону  -  1.1584519420739245
    дополнительный  -  0.9412870315038989
    допускать  -  2.7482250447558583
    достижение  -  0.9251597658713901
    достоверность  -  1.7232993061362278
    достойный  -  1.2004899560319777
    доходить  -  0.790862318948725
    доходчиво  -  0.8874329362937291
    евгеньевна  -  1.3896775193932165
    его  -  0.6339368096542006
    егрюл  -  0.9588172734625438
    ежемесячный  -  0.6453501924872029
    езжать  -  0.792718039624636
    екб  -  0.6289250010763889
    екоммерц  -  0.9289793708866114
    есиа  -  0.6403126819315391
    естественно  -  0.5901840979233905
    ецд  -  0.9775052225428151
    ждемс  -  1.4054657752475275
    желательно  -  1.4873180738398826
    жизненный  -  1.5319039009245632
    житель  -  0.6765880326976143
    заблокировать  -  1.6088756071896442
    забота  -  1.1410556468173787
    заведомо  -  0.6453501924872029
    зависеть  -  0.8227933123442936
    заводить  -  0.9172376470694468
    загружаться  -  1.3178753210123908
    загрузка  -  0.6794626071505669
    задавать  -  1.1584228147802207
    заданный  -  0.9005584242037462
    задержка  -  1.5237804714806957
    задолжать  -  1.576074817040058
    задолжник  -  0.8288847293580721
    задолжность  -  0.7882482119130646
    заемщик  -  2.4092344728747923
    зайти  -  3.3640163318245495
    заказной  -  0.6742677758886515
    заказывать  -  1.7257663640092002
    заключать  -  0.6179826961949022
    заключаться  -  0.8227933123442936
    закон  -  0.6044039086200238
    законспирировать  -  2.66230492997297
    закрывать  -  0.754844597214708
    закрытие  -  0.6727077884750141
    занятый  -  0.5977660343596116
    зао  -  0.9319080485932456
    записываться  -  0.6809466712221353
    заплатить  -  1.540621980992455
    заполняться  -  0.6179826961949022
    запонивать  -  0.6491828206830214
    запрет  -  0.9242052585162236
    запрос  -  0.8342409102694774
    запускать  -  0.6727077884750141
    заранее  -  0.7964047875092225
    зарегистрировать  -  1.0937527847892736
    застраховывать  -  2.4128043547219944
    затем  -  1.8033384613664685
    затираться  -  0.8558901313833194
    зачем  -  1.0893218573475698
    защита  -  1.0653062896505354
    заявка  -  0.9902992239211004
    звонок  -  0.8620140451820719
    здесь  -  1.2279160952855133
    здравствовать  -  0.6179195187558563
    здравствоватьу  -  0.6453501924872029
    здравстовать  -  1.3896775193932165
    знакомый  -  0.6006664362293961
    значиться  -  1.0115781105633757
    зубцовский  -  0.9304858965703752
    иго  -  0.6727077884750141
    идеально  -  0.7796624040724537
    идея  -  1.0693203256545225
    извинять  -  0.6727077884750141
    изменение  -  1.718551971786471
    изменять  -  0.9656136629619769
    изменяться  -  0.93077706350563
    изымать  -  0.8660265166246273
    иконка  -  0.9670187052409837
    индекс  -  0.6544950585302476
    индикатор  -  0.856654375942677
    инженерно  -  1.057483651888801
    инн  -  1.8622638520426835
    иностранец  -  0.6727077884750141
    инспектор  -  0.7884297187745462
    институт  -  0.6428745794120495
    интересно  -  0.8230192788773961
    интересный  -  1.024042282774779
    интересовать  -  3.0656416434621665
    инфо  -  0.7768882409200175
    информативный  -  1.170955410037328
    иркутск  -  1.8308258203282581
    иркутский  -  0.9154129101641291
    искать  -  1.188394260875656
    исполнитель  -  1.840213498721822
    исполнительный  -  0.6529918189015522
    исполняться  -  0.6160865356287717
    использование  -  0.6195653874623899
    истечение  -  0.7694456067492205
    источник  -  0.9775052225428151
    исх  -  1.057483651888801
    июнь  -  0.8791736696384282
    ия  -  0.9603279301848189
    кабинет  -  0.884551081788833
    кадры  -  1.5946244548867035
    календарь  -  0.632432356779463
    калькулятор  -  1.8648978161016767
    канализационный  -  0.9666481230013241
    капитал  -  2.6707316594209587
    капитальный  -  0.9666481230013241
    капча  -  0.8350567428648374
    картина  -  1.260854731838191
    картинка  -  2.3928790282718486
    карточка  -  1.2683856336909258
    кв  -  0.6277495963834032
    квартира  -  1.4575357456031464
    квота  -  1.0343256075171003
    кех  -  0.9289793708866114
    клевета  -  0.7155484663895009
    ключевой  -  0.9005584242037462
    кнопка  -  0.9935812066578307
    код  -  1.4325681401417851
    кока  -  1.0326261242896069
    коллегия  -  0.8558901313833194
    колонка  -  0.5922556089308869
    комбинация  -  0.8895460339273343
    комиссия  -  1.9628370308557965
    комментарий  -  0.7070057162854264
    коммунальный  -  1.3134778239837557
    комплекс  -  0.8384957195784459
    кондрово  -  0.6133708182546562
    конец  -  1.2713690033348453
    конкретизировать  -  1.057483651888801
    конкретно  -  0.6453501924872029
    конкурс  -  0.7784049899072009
    консультирование  -  0.8843970758833342
    контакт  -  1.9300613875616124
    контактный  -  0.5901840979233905
    копия  -  2.6243135950780214
    короткова  -  0.811894221023322
    корректировка  -  0.6014943929571288
    корректно  -  2.404048743644573
    краснодар  -  1.057483651888801
    краснофлотский  -  0.5901840979233905
    красноярский  -  2.314730745232325
    красочный  -  1.1305684197207158
    краткость  -  0.8464546734901707
    кредит  -  1.0002675448826095
    кропотливый  -  0.6394938440559914
    крутиться  -  0.7314877989544122
    крымчанин  -  1.1979243467614928
    кстати  -  1.0504386548621119
    культура  -  0.6842804773738892
    курский  -  1.0382506485079155
    лаборатория  -  0.8548743674304884
    лавров  -  0.7716227138425236
    лазить  -  1.039746425192507
    легко  -  1.2172411893824202
    ленинградский  -  1.747364601086053
    леонид  -  1.747364601086053
    лето  -  0.6013596942105434
    либо  -  1.0545065942199852
    линия  -  0.7106918605080919
    лист  -  0.595959484880759
    личной  -  1.0611661353605495
    личный  -  0.6168945124697783
    лишать  -  1.0326261242896069
    лишаться  -  1.241388012772533
    лишний  -  0.6580056328480555
    лк  -  0.7069552619528395
    лог  -  2.4454316724174734
    май  -  1.0393745056071064
    максимум  -  1.260854731838191
    мало  -  4.159658123107802
    малый  -  0.6216823874316262
    материнский  -  2.167455913888957
    материснком  -  0.6855924976432219
    медицинский  -  1.4884427447964583
    медленно  -  1.230574665333181
    междуреченск  -  3.6069373933280704
    меняться  -  0.9069512806537388
    местный  -  0.6778201605161451
    мешать  -  0.7786325549137884
    ми  -  0.7786325549137884
    миасс  -  0.6289250010763889
    миллион  -  1.260854731838191
    минфин  -  0.9450987256622317
    минэнерго  -  2.0912310013711126
    мипорт  -  0.5922556089308869
    мировой  -  0.7790877819650697
    михаил  -  0.6934144288746056
    михайлович  -  1.208136779058019
    многое  -  0.6259505807890436
    многочисленный  -  0.6727077884750141
    мобильный  -  0.7050571274508728
    модальный  -  0.6160865356287717
    можжно  -  1.6088756071896442
    можно  -  4.004603668527839
    мой  -  1.5384451492391884
    молодец  -  1.282104928366779
    молодой  -  1.0855716132512727
    мониторинг  -  0.6363670728155948
    моорм  -  1.5450450450450446
    мотивировать  -  0.7443111557909914
    муп  -  0.9154129101641291
    мурманский  -  0.8666898741079008
    навешивать  -  0.7786325549137884
    навигация  -  0.9387066923223384
    наглядный  -  1.1305684197207158
    нагрузка  -  2.06854829362885
    надеяться  -  1.5863535921887288
    надпись  -  0.6006664362293961
    нажатие  -  0.8119090807366665
    назад  -  2.480899274840686
    наиболее  -  1.0456155006855563
    наконец  -  1.135981353181101
    наличие  -  2.6324043532933805
    налог  -  1.1511516201862162
    налогоплательщик  -  0.6027063477081389
    намерен  -  0.7155484663895009
    напечатать  -  1.615986221232263
    например  -  1.512203248695519
    нарисоваться  -  1.6071141826448447
    наркотический  -  0.6088640298305359
    народность  -  1.8430105255424647
    нарушаться  -  0.7042082950267005
    население  -  0.8302609837704635
    населенный  -  0.8203346076293703
    настраивать  -  0.9491989144399908
    наступление  -  0.8558901313833194
    национальность  -  1.8430105255424647
    наш  -  0.949977597606264
    неактуальный  -  0.6179826961949022
    неверный  -  1.4770396748237857
    невозможно  -  1.4135360515708983
    негатив  -  0.5901840979233905
    неграмотный  -  0.6727077884750141
    недавний  -  1.241388012772533
    недвижимость  -  0.8038342520281667
    неделя  -  0.8593378178319317
    недопустимо  -  0.658853722343447
    недоступный  -  1.7368917010237357
    ненужный  -  2.883629112807295
    необычайно  -  0.7786325549137884
    неодкратно  -  0.7982554366152912
    неоднозначный  -  0.6160865356287717
    неоднократный  -  0.9603279301848189
    неплохой  -  1.715549052627943
    неполный  -  0.7662784451342981
    непонятно  -  0.9172376470694468
    непостижимый  -  0.9964814341148005
    неправда  -  0.9656136629619769
    неразрешение  -  0.856654375942677
    нереально  -  1.1378595325446186
    несмочь  -  3.078250140139968
    несоответствующий  -  1.1378595325446186
    нетерпение  -  0.6727077884750141
    нету  -  0.7008053236005517
    нигде  -  1.0445663718026057
    никак  -  0.7301643179631538
    никакой  -  1.1387166975412386
    николаевич  -  1.747364601086053
    николаевна  -  0.8876452594355744
    николай  -  1.316749879950323
    никто  -  0.8912209125743982
    новостной  -  0.6160865356287717
    ноль  -  1.10652919336721
    номер  -  1.9533711396014788
    норматив  -  0.9117584697629896
    нормативный  -  0.7314558674431636
    нпо  -  0.9319080485932456
    нпф  -  0.7154263750164873
    нравиться  -  1.2965373403162954
    нужный  -  0.8143992708695378
    оао  -  0.9666481230013241
    обеовляться  -  1.190800657588635
    облэнерго  -  0.792718039624636
    обнаружение  -  0.7768882409200175
    обнаруживать  -  1.132612327956877
    обновление  -  0.8203940070503895
    обновлять  -  0.9139963154101536
    обновляться  -  1.1919783860352329
    обозначать  -  0.6453501924872029
    обосновывать  -  1.0100826860774303
    обратно  -  0.7154263750164873
    обратный  -  0.7765887796904793
    обращать  -  0.6942343192498835
    обращение  -  1.6421420663927428
    обстоять  -  1.747364601086053
    общий  -  0.7671629919362934
    объеденить  -  1.9165737468159192
    объявление  -  0.8558901313833194
    объявлять  -  0.8075344861535738
    обычно  -  0.6373680618427552
    оглавление  -  1.1378595325446186
    ограничение  -  0.601079824996388
    огрнип  -  0.6143946428028509
    огромный  -  0.6777299533963175
    однако  -  0.9775136951861398
    оказание  -  1.4884427447964583
    оказываться  -  0.6686784069665817
    окно  -  0.6948506086903354
    окончание  -  0.7662784451342981
    ольга  -  0.811894221023322
    омский  -  0.7352579448488498
    онлайн  -  0.8422311529259945
    ооо  -  2.7375596743300066
    опасный  -  0.8203346076293703
    описывать  -  0.7155484663895009
    оплата  -  0.6440635424656288
    оплачивать  -  1.0875879271441702
    оповещать  -  0.8288847293580721
    определение  -  0.7790877819650697
    опровержение  -  0.7155484663895009
    опрос  -  1.098332960172843
    оптимизировать  -  1.3890842200997342
    опубликовывать  -  1.1018390852301296
    опыт  -  0.758097414589642
    опять  -  1.1064031980445554
    оранжевый  -  0.6160865356287717
    организация  -  1.512903150494433
    основание  -  1.6784288179309832
    основной  -  0.7789856074437755
    осп  -  1.4404390033770222
    осуществляться  -  1.3927926029217021
    ответственнен  -  0.6453501924872029
    ответственно  -  0.5901840979233905
    отвечать  -  1.6973451619350592
    отдельно  -  0.7823829339831326
    отдельный  -  3.738643118264941
    отзывчивость  -  1.5933387380324213
    отказ  -  2.0201653721548607
    отказываться  -  0.8561168600208773
    открывать  -  0.6457170579971109
    открываться  -  0.6355137587934131
    отличный  -  0.9657977918794156
    отмена  -  1.0100826860774303
    отменять  -  0.8765100138631854
    относиться  -  1.0343059790307956
    отношениепричем  -  0.6453501924872029
    отображение  -  0.9896677783893539
    отопление  -  1.3134778239837557
    отправка  -  1.7581569637057783
    отправление  -  0.6013596942105434
    отправлять  -  1.0476620091951303
    отражать  -  1.71145451107028
    отслеживать  -  0.609967238671625
    отстутствовать  -  1.186149384684255
    отсылать  -  0.5915741653339025
    отходы  -  0.9179075353892274
    отчет  -  0.777872297942447
    отчетность  -  0.8485031823434943
    офис  -  0.7389459882067753
    официальный  -  1.3455950283878104
    оформлять  -  0.9839872780587701
    очевидно  -  0.867466287491473
    очень  -  2.004257139885737
    очередной  -  0.8118942292687301
    ошибка  -  0.6595574885247838
    паспорт  -  1.3620594179723264
    паспортизация  -  0.9179075353892274
    пд  -  0.9603279301848189
    пенсионер  -  0.9639891623811953
    пенсия  -  1.252910055290429
    первоначально  -  0.7197412809889817
    первоначальный  -  0.9964814341148005
    переадресовывать  -  0.8739323357054329
    передавать  -  0.8316922847019822
    переезд  -  1.0829015100808452
    перезагружаться  -  1.1380137598411575
    перенести  -  0.6641468841834922
    перепуг  -  0.7304374451290715
    пересылка  -  0.9372524296015834
    переход  -  1.110851745463123
    перечень  -  0.8854197298060272
    перечислять  -  1.8051055831080984
    период  -  0.8742771033111534
    периодически  -  0.6453501924872029
    пестрый  -  0.6160865356287717
    печать  -  1.8011153498254286
    печерина  -  3.6069373933280704
    планировать  -  0.8726977679775523
    планироваться  -  1.241388012772533
    платеж  -  1.5582923976588945
    платежок  -  0.5887261690118823
    плательщик  -  1.183663993456408
    плашка  -  2.66230492997297
    плейер  -  0.6195207803956226
    плохо  -  1.405736520357494
    поверочный  -  1.5578286081368793
    повышение  -  1.219227162596113
    поганый  -  0.8660265166246273
    погода  -  0.8203346076293703
    подача  -  0.608969785432254
    подключать  -  0.601464905109542
    подробно  -  0.7598462276018135
    подсказывать  -  1.9637752358093152
    подтв  -  0.6565878183428813
    пожалуйста  -  0.8949628332618981
    позволять  -  1.47851591416659
    поинтересоваться  -  0.7066378950830271
    поискать  -  0.590281554878276
    поисковик  -  0.9491989144399908
    пока  -  0.6915338934970346
    покрытие  -  0.9666481230013241
    поле  -  0.7630429168129891
    полномочие  -  0.6148389869427752
    полностью  -  0.7479538543091346
    половина  -  1.5449347731075844
    положительно  -  0.6289250010763889
    получатель  -  1.216679050699732
    получаться  -  1.2377954173102266
    получение  -  1.1817207523441886
    полученый  -  0.6386812181915693
    поляков  -  0.7790877819650697
    помощь  -  1.1795388496104366
    понимать  -  1.1972556739841718
    понравиться  -  2.1107552326782266
    понятно  -  0.6637308165337904
    понятный  -  1.038970631925124
    попадать  -  0.6004703027039257
    портал  -  0.8502803492622344
    поручение  -  0.9034160514169277
    посмотреть  -  1.281260303646869
    поставлять  -  0.7253832379284699
    поставщик  -  1.8308258203282581
    построение  -  0.9491989144399908
    потеря  -  0.6160865356287717
    потому  -  0.7149299801994217
    поход  -  1.0936664421851297
    почему  -  2.0537552821855067
    починять  -  0.6418403128716426
    почитать  -  0.868076908776531
    почти  -  0.6338142652962755
    почтовый  -  1.1569651900463171
    пошлина  -  0.9372524296015834
    появляеться  -  1.713308751885354
    ппри  -  0.856654375942677
    правильно  -  0.6254752677843131
    правильность  -  1.2298826969208567
    правительство  -  0.8575711552945462
    правонарушение  -  1.3491430868262553
    правый  -  0.6160865356287717
    практически  -  0.6215043300405523
    практичный  -  0.6329575767608356
    преградный  -  0.6544950585302476
    предварительный  -  0.6240212555051845
    предел  -  0.6350603809574168
    предистория  -  0.7155484663895009
    предлагать  -  0.8279978630700011
    предлагаться  -  0.6298491186909355
    предложение  -  0.6459264904680282
    предоставлять  -  0.8407534247250563
    предприниматель  -  0.590281554878276
    предстоящий  -  0.8558901313833194
    предыдущий  -  0.657447675732528
    препятствовать  -  0.6006664362293961
    при  -  1.141884657403085
    привлекать  -  0.6179826961949022
    приводить  -  2.600984600547922
    придумывать  -  0.6148225304499055
    прием  -  1.437068555069123
    прикладывать  -  0.891651593170433
    принудительно  -  0.8551787593331528
    приобретать  -  0.6006664362293961
    природопользователь  -  0.595938328290359
    пристав  -  0.8529458586518066
    присылать  -  0.687893362342421
    приходиться  -  0.7236514841180088
    причем  -  1.4712279589615191
    про  -  0.6039516892496279
    проблема  -  0.8261618259026238
    проблематичный  -  0.9034160514169277
    провайдер  -  1.150494487697588
    проверт  -  1.2298826969208567
    прогноз  -  0.6363670728155948
    программный  -  0.787932083834448
    продлять  -  0.9500439774953672
    проект  -  0.8182719063748151
    производить  -  0.8446660386080869
    производство  -  0.6837558337528046
    произвольный  -  1.1378595325446186
    промышленный  -  0.792718039624636
    пропускать  -  0.8558901313833194
    просить  -  0.7942981898253927
    просматривать  -  0.7253832379284699
    простой  -  0.739084330727327
    просьба  -  1.2699160537229401
    протяжение  -  0.811894221023322
    проффесий  -  1.8375777822151649
    проходить  -  1.2599324377358208
    прямо  -  1.011067394780822
    прямоугольник  -  0.6160865356287717
    пункт  -  0.8450916654904385
    пф  -  0.9798089027602102
    пфр  -  2.7717859237583093
    пятый  -  0.9289793708866114
    работа  -  1.5187045581992482
    работать  -  1.1435780067646149
    работник  -  1.0098777392427518
    работодатель  -  1.2359653923898044
    рабочий  -  0.7962388451151341
    равняться  -  0.6010826979029216
    развиваться  -  0.7155484663895009
    развод  -  1.2452270446877933
    разводить  -  0.792718039624636
    размер  -  1.5195195206457188
    разработчик  -  2.1559555799972916
    разрешать  -  0.8464546734901707
    разрешение  -  0.7174637475103534
    разумный  -  0.6437077557036678
    разъяснение  -  0.8734095479841266
    разъяснять  -  0.710841112428459
    район  -  1.0646350077399034
    ранее  -  1.1305684197207158
    раскрывать  -  1.492803922812186
    распечатка  -  1.0246464011348055
    распечатываться  -  1.0777751277605125
    распределять  -  1.5946244548867035
    распространение  -  0.6088640298305359
    рассматривать  -  2.9188157770225853
    рассматриваться  -  1.0010465601391936
    рассмотрение  -  1.7533843342467428
    расход  -  0.9666481230013241
    расчет  -  0.9807611677285426
    реагировать  -  0.6852343209754712
    реакция  -  1.3704239950589858
    реализация  -  0.7622232179421251
    реально  -  1.3354786512180685
    регион  -  3.5870612238844366
    регламент  -  3.045276374085054
    регулярно  -  2.4070504117878073
    редко  -  0.8415327585870362
    режим  -  0.7518579700061803
    рез  -  0.9603279301848189
    резидент  -  0.7557820933042203
    результирующий  -  0.5922556089308869
    реквизит  -  0.8559178399812905
    республика  -  0.9122905597648129
    ресурс  -  1.2328221620458528
    робокасса  -  1.2452270446877933
    розыск  -  1.0520623675135696
    роль  -  0.6403126819315391
    росаккредитация  -  0.8850988363336068
    роспотребнадзор  -  0.8570841875505094
    российский  -  0.7402224698476454
    ростов  -  1.1584519420739245
    рубль  -  1.813790683082402
    рубльпочему  -  0.6453501924872029
    рубрика  -  2.559908823457091
    рука  -  1.2061689554800652
    рынок  -  0.6363670728155948
    сазонова  -  0.7790877819650697
    самый  -  1.450840873852662
    сараджев  -  0.6934144288746056
    сбербанк  -  1.1324209546074968
    сбой  -  0.7194343675593462
    сбор  -  1.3563789869226852
    свидетельство  -  0.9762686977450215
    сводный  -  0.7317670253010105
    свыше  -  1.260854731838191
    сд  -  0.8371860821345405
    сдавать  -  2.485871742736276
    сдвигаться  -  0.5922556089308869
    сделка  -  0.7100389318412598
    севастополь  -  2.5638465528951344
    сервис  -  0.6929888313476926
    сергеевич  -  1.576074817040058
    середина  -  1.615986221232263
    сертификат  -  0.593115619643199
    серый  -  1.2321730712575434
    серьезный  -  0.8464546734901707
    сессия  -  2.4454316724174734
    сеть  -  1.6880841605872883
    символ  -  0.8047437455072348
    системный  -  1.0935980130251781
    ситуация  -  1.2538285844414885
    скатить  -  1.2298826969208567
    скачать  -  2.1442819093061902
    скачиваться  -  1.260854731838191
    скашивать  -  0.6160865356287717
    скорый  -  0.9304858965703752
    слева  -  0.7786325549137884
    след  -  0.7768882409200175
    следовать  -  0.6712485189257877
    служащий  -  0.6842804773738892
    случай  -  0.5978871737012882
    слушать  -  0.792718039624636
    смешно  -  2.3928790282718486
    смеяться  -  0.8448217956832408
    смотреть  -  0.9664209755616009
    смп  -  0.834991143517642
    смс  -  0.6267350949492793
    смысл  -  1.2006431623715306
    снижать  -  1.260854731838191
    снизу  -  0.7786325549137884
    снилс  -  1.347588812261239
    снимать  -  1.4856896859348792
    снятие  -  0.6289250010763889
    советский  -  0.6131430852553366
    совхоз  -  1.4980263435779813
    содержательный  -  0.8977349925302183
    соединение  -  1.3059230539999893
    создавать  -  1.2667589706257676
    создание  -  0.8566723374398104
    сокращать  -  0.8464546734901707
    сомнение  -  0.8436568581182602
    сообщение  -  1.5756119540609328
    соответсвий  -  0.6386812181915693
    состояние  -  0.6071083254906895
    состояться  -  0.8558901313833194
    сотрудник  -  1.6546290322262498
    сотый  -  2.3928790282718486
    социальный  -  1.6508928985201266
    сочетание  -  0.6160865356287717
    спасибо  -  5.248539980903813
    спать  -  1.5160626586096355
    специализированный  -  0.7042082950267005
    специалист  -  0.8967611853729323
    специально  -  2.3928790282718486
    специальный  -  1.621179466284898
    список  -  0.615876225336923
    списываться  -  0.8551787593331528
    спланировать  -  0.792718039624636
    спо  -  0.7768882409200175
    справка  -  0.6791738911484905
    спрашивать  -  0.6013596942105434
    спутник  -  0.8203346076293703
    сравнение  -  0.7878306373623636
    средство  -  0.6134402393013907
    срок  -  1.0956797011167947
    стаж  -  0.7731453713549203
    стандарт  -  2.4660611448471177
    старинка  -  0.8149409368739463
    стартовый  -  0.8558901313833194
    старый  -  0.678125310138604
    статус  -  0.6140282118308046
    статья  -  1.12321013544068
    стоить  -  1.6928046934708099
    стол  -  0.6525178175788067
    сторона  -  0.8591107649830186
    страница  -  0.6251296287339642
    страховой  -  0.9691474167846806
    строительство  -  0.9666481230013241
    строка  -  3.424627163094065
    структурный  -  0.6198437813019253
    суд  -  0.7681849643011378
    судья  -  0.7790877819650697
    сумма  -  2.0927507889732793
    сутки  -  0.9804964092478671
    существенно  -  1.260854731838191
    табличка  -  0.7544755123228725
    так  -  0.6171865752859246
    таможенный  -  1.722835371210327
    тариф  -  0.6957245789000392
    тверская  -  0.9304858965703752
    твитнуть  -  0.8464546734901707
    твитов  -  0.8464546734901707
    твиттер  -  0.8464546734901707
    текст  -  0.8919859912832879
    текстильщик  -  0.9775052225428151
    текучка  -  1.5946244548867035
    текущий  -  0.9666481230013241
    телефон  -  3.8967575755261246
    телефонный  -  1.640306068540717
    тело  -  1.6558769140479224
    теперь  -  0.8134331440189169
    терпение  -  0.6394938440559914
    территориальный  -  1.1766223952540211
    территория  -  0.5964866978986226
    терять  -  0.7309553820153842
    теряться  -  0.9451623248608234
    типовой  -  1.0456155006855563
    тишник  -  0.6348982699513213
    тнвэд  -  2.521709463676382
    товар  -  0.5988065230684658
    товарный  -  0.6363670728155948
    тогда  -  0.6013596942105434
    толк  -  1.4998571278905923
    только  -  1.2285854186714902
    толя  -  0.7112043150258175
    тонкий  -  0.6160865356287717
    тот  -  0.5997775887118653
    транслировать  -  0.7749638070618856
    трансляция  -  0.6195207803956226
    транспортный  -  1.4950642934971128
    траспортогой  -  0.6386812181915693
    требование  -  0.773034463688389
    трекинговый  -  2.4454316724174734
    третий  -  0.6013596942105434
    три  -  1.0903761346575513
    трубка  -  0.6287158545173404
    трудоустройство  -  0.6778201605161451
    тыс  -  1.576074817040058
    тысяча  -  2.527617469665895
    убирать  -  3.385811504299664
    увеличиваться  -  0.6453501924872029
    увидеть  -  0.8217519381917802
    увлечение  -  0.6160865356287717
    удалять  -  1.2964306660425782
    удерживать  -  1.2374544684526256
    удивленный  -  0.6453501924872029
    удивляться  -  0.6289250010763889
    удобный  -  3.3289957314791074
    удобство  -  2.853292174967941
    удостоверение  -  0.6304207743299499
    ужас  -  0.9605975738492016
    уип  -  0.6544950585302476
    ук  -  0.7155484663895009
    указывать  -  2.3995020911725002
    указываться  -  1.576074817040058
    ул  -  0.9936023236552024
    улучшение  -  0.6008894225757322
    уменьшаться  -  3.4091015859600176
    уплата  -  0.9034160514169277
    управление  -  0.8232688621087055
    уровень  -  0.686672945730225
    урупский  -  0.8726600780403302
    услуга  -  0.8470691683754141
    услышать  -  1.676753970811102
    усовершенствование  -  1.260854731838191
    успешный  -  1.1380137598411575
    утверждать  -  0.7724177596404642
    утеря  -  0.6821821002523207
    утерять  -  0.6614062840644256
    уфмс  -  1.4044179783946602
    участвовать  -  1.241388012772533
    участок  -  0.7796624040724537
    фадеев  -  2.475416635663559
    фактически  -  0.8551787593331528
    февраль  -  1.0282149250820294
    федерация  -  1.2087320276305293
    фильтр  -  1.8011168484074924
    фильтрация  -  1.9140459814797464
    фипс  -  0.5971501173822538
    фл  -  0.7197412809889817
    флет  -  0.7318218344706215
    форма  -  0.9744431308330332
    формат  -  0.7634922052642615
    формироваться  -  0.6473111888721235
    форум  -  1.1656116276187052
    фотоснимок  -  0.8203346076293703
    фраза  -  1.6030562517642997
    фссп  -  2.655301827116685
    функционал  -  0.6160865356287717
    хабаровск  -  0.7785672286899396
    хамракулов  -  1.3896775193932165
    хватать  -  1.0677927592628595
    химия  -  1.7440985826554554
    холодный  -  0.9154129101641291
    хотеться  -  0.8407021494509298
    хоть  -  0.8448217956832408
    централизованный  -  0.9666481230013241
    частность  -  1.195300868619602
    часы  -  1.123480831315759
    чей  -  1.3890842200997342
    чело  -  0.6078389798419932
    человек  -  0.9855451825593133
    чем  -  1.4800844804313655
    через  -  1.4396593398617905
    числиться  -  1.3819954980350913
    читабельный  -  0.6160865356287717
    читать  -  1.497051119559124
    читаться  -  0.7317670253010105
    чтение  -  0.8234240762309322
    что  -  1.8949194968656649
    чтоб  -  1.0295518310621299
    чуть  -  0.8558901313833194
    шаг  -  0.6473688351727886
    экземпляр  -  0.6179826961949022
    электроэнергия  -  0.792718039624636
    элемент  -  0.9289793708866114
    этопосмотреть  -  0.6307895887708203
    этот  -  1.028670781968208
    юбилейный  -  0.9930272007770017
    юл  -  0.7146148161735411
    юр  -  0.9603279301848189
    юрий  -  0.605372143829737
    явление  -  0.8203346076293703
    язык  -  0.8326139319721104
    ялта  -  0.5915741653339025
    январь  -  0.9366512850045405
    


```python
# фичи лежат в data
# ответы в YY
# Проведем разделение текстов на две части
# импортируем поцедуру деления датетса на тестовую и обучающую коллекцию.
from sklearn.model_selection import train_test_split
# Делим нашу клддекцию на две части: 1. часть для обучения 2. часть для проверки результатов обучения
X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.33, random_state=42)
print('Test collection size: ', X_test.shape)
print('Training collection size: ', X_train.shape)
```

    Test collection size:  (187, 1000)
    Training collection size:  (378, 1000)
    

Обучаем модели: 


```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn import metrics
## Import the Classifier.
from sklearn.neighbors import KNeighborsClassifier
# устанавливаем число соседей
n_neighbors=1

# тип взвешивания: ‘uniform’ : uniform weights, ‘distance’ : weight points by the inverse of their distance
weights = 'uniform'

# тип расстояния (p = 1 - manhattan_distance, p = 2 - euclidean_distance, )
p = 2

# автоматический подбор типа алгоритма
algorithm = 'auto'

#Создаем модель
knn = KNeighborsClassifier(n_neighbors, weights, algorithm, p)

## Обучаем модель на тренироваочных данных
knn.fit(X_train, y_train)

# оцениваем точность нашей модели на основе известных значений y_train
predicted = knn.predict(X_test)

# расчитываем основные метрики качества
print(metrics.classification_report(y_test, predicted))

print('confusion matrix')
print(metrics.confusion_matrix(y_test, predicted))
```

                  precision    recall  f1-score   support
    
              -1       0.46      0.71      0.56        70
               0       0.64      0.41      0.50        39
               1       0.68      0.46      0.55        78
    
        accuracy                           0.55       187
       macro avg       0.59      0.53      0.54       187
    weighted avg       0.59      0.55      0.54       187
    
    confusion matrix
    [[50  6 14]
     [20 16  3]
     [39  3 36]]
    


```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
neighbors=10
weights = 'uniform'
p = 2
#организуем пустой массив scores
cv_scores_train = []
cv_scores_test = []


# в цикле перебираем число соседей и расчитываем среднее значение F1
for k in range(1, neighbors):
    knn = KNeighborsClassifier(k, weights, algorithm, p)
    knn.fit(X_train, y_train)
    # knn.score(X_test, y_test)
    # подсчет среднего значения F1 при разных разбиениях
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='f1_macro')
    # расчет среднего значения scores по всем разбиениям
    #print(np.mean(scores))
    cv_scores_train.append(np.mean(scores))
    
    scores = cross_val_score(knn, X_test, y_test, cv=5, scoring='f1_macro')
    # добавляем в список очереное среднее значение f1_macro
    cv_scores_test.append(np.mean(scores))
   
    # вывод среднего значения F1 по всем классам
    #print(f1_score(y_test, knn.predict(X_test), average='macro'))
    # расчитываем F1 и добавляем его в список
    #cv_scores_test.append(f1_score(y_test, knn.predict(X_test), average='macro'))

# строим график F1
plt.plot(cv_scores_train, 'ro', alpha=0.2)
plt.plot(cv_scores_train, '-bo', linewidth=1)

plt.plot(cv_scores_test, 'ro', alpha=0.2)
plt.plot(cv_scores_test, 'r', alpha=0.2)
plt.show()    
print('расчет закончен')
#1 сосед топ
```

    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    


![png](output_9_1.png)


    расчет закончен
    


```python
n_neighbors=2

# тип взвешивания: ‘uniform’ : uniform weights, ‘distance’ : weight points by the inverse of their distance
weights = 'uniform'

# тип расстояния (p = 1 - manhattan_distance, p = 2 - euclidean_distance, )
p = 2

# автоматический подбор типа алгоритма
algorithm = 'auto'

#Создаем модель
knn = KNeighborsClassifier(n_neighbors, weights, algorithm, p)

## Обучаем модель на тренироваочных данных
knn.fit(X_train, y_train)

# оцениваем точность нашей модели на основе известных значений y_train
predicted = knn.predict(X_test)

# расчитываем основные метрики качества
print(metrics.classification_report(y_test, predicted))

print('confusion matrix')
print(metrics.confusion_matrix(y_test, predicted))
```

                  precision    recall  f1-score   support
    
              -1       0.40      0.90      0.55        70
               0       0.67      0.10      0.18        39
               1       0.70      0.21      0.32        78
    
        accuracy                           0.44       187
       macro avg       0.59      0.40      0.35       187
    weighted avg       0.58      0.44      0.38       187
    
    confusion matrix
    [[63  2  5]
     [33  4  2]
     [62  0 16]]
    

KNN не оч


```python
# импортируем модель
from sklearn import metrics
from sklearn.svm import SVC



# определяем модель классификатора
model = SVC()
# обучаем модель на тренировочном датасете
model.fit(X_train, y_train)
# выводим на экран параметры модели.
print(model)

# строим предсказание
predicted = model.predict(X_test)

# определяем качество модели
print(metrics.classification_report(y_test, predicted))
# строим confusion matrix
print(metrics.confusion_matrix(y_test, predicted))
```

    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\svm\base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    

    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
        kernel='rbf', max_iter=-1, probability=False, random_state=None,
        shrinking=True, tol=0.001, verbose=False)
                  precision    recall  f1-score   support
    
              -1       0.37      1.00      0.54        70
               0       0.00      0.00      0.00        39
               1       0.00      0.00      0.00        78
    
        accuracy                           0.37       187
       macro avg       0.12      0.33      0.18       187
    weighted avg       0.14      0.37      0.20       187
    
    [[70  0  0]
     [39  0  0]
     [78  0  0]]
    

    C:\Users\SOSI\Anaconda3\lib\site-packages\sklearn\metrics\classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    


```python
from sklearn import svm
from sklearn.model_selection import GridSearchCV

param_grid = [
  {'C': [1, 10], 'kernel': ['linear']},
  {'C': [1, 10], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

svc = svm.SVC(gamma="scale")
clf = GridSearchCV(svc, param_grid, cv=5)
clf.fit(X_train, y_train)
# выводим наилучший результат
clf.best_params_

```


```python
# определяем модель классификатора
model = SVC(C=10, kernel='linear')
# обучаем модель на тренировочном датасете
model.fit(X_train, y_train)
# выводим на экран параметры модели.
print(model)

# строим предсказание
predicted = model.predict(X_test)

# определяем качество модели
print(metrics.classification_report(y_test, predicted))
# строим confusion matrix
print(metrics.confusion_matrix(y_test, predicted))
```

    SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
        kernel='linear', max_iter=-1, probability=False, random_state=None,
        shrinking=True, tol=0.001, verbose=False)
                  precision    recall  f1-score   support
    
              -1       0.64      0.86      0.73        70
               0       0.88      0.36      0.51        39
               1       0.74      0.73      0.74        78
    
        accuracy                           0.70       187
       macro avg       0.75      0.65      0.66       187
    weighted avg       0.73      0.70      0.69       187
    
    [[60  1  9]
     [14 14 11]
     [20  1 57]]
    

SVC ниче так 53


```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(
    n_estimators=50,
    criterion='gini',
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='auto',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    random_state=0,
    verbose=0,
    warm_start=False,
    class_weight='balanced')


clf.fit(X_train, y_train)
# строим предсказание
predicted = clf.predict(X_test)

# определяем качество модели
print(metrics.classification_report(y_test, predicted))
# строим confusion matrix
print(metrics.confusion_matrix(y_test, predicted))
```


```python
param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

clf = RandomForestClassifier(
    n_estimators=50,
    criterion='gini',
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='auto',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    random_state=0,
    verbose=0,
    warm_start=False,
    class_weight='balanced')

clf = GridSearchCV(estimator= clf, param_grid=param_grid, cv= 5, scoring='f1_macro')
clf.fit(X_train, y_train)
# выводим наилучший результат
print('best parameters: ', clf.best_params_)
```


```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(
    n_estimators=500,
    criterion='entropy',
    max_depth=8,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='auto',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=-1,
    random_state=0,
    verbose=0,
    warm_start=False,
    class_weight='balanced')


clf.fit(X_train, y_train)
# строим предсказание
predicted = clf.predict(X_test)

# определяем качество модели
print(metrics.classification_report(y_test, predicted))
# строим confusion matrix
print(metrics.confusion_matrix(y_test, predicted))
```

                  precision    recall  f1-score   support
    
              -1       0.53      0.56      0.54        70
               0       0.82      0.36      0.50        39
               1       0.57      0.71      0.63        78
    
        accuracy                           0.58       187
       macro avg       0.64      0.54      0.56       187
    weighted avg       0.61      0.58      0.57       187
    
    [[39  2 29]
     [13 14 12]
     [22  1 55]]
    

НЕПЛОХО 52



```python

```


```python
# и так, мы обучили классификатор KNN (для простоты не делал поиск оптимального числа соседей)
#  теперь читаем новые данные, по которым нужно сделать предсказание
 #читаем новые данные 
f = 'C:/Users/SOSI/Downloads/data_new.csv'
df1=pd.read_csv(f,  sep=';', encoding='utf8')
```


```python
# this gives us the size of the array
print('Dataset size', df1.shape)

# here we can get size of array as two variables
num_rows1, num_feature1 = df1.shape

print('row number: ', num_rows1)
print('feature number: ', num_feature1)
print('names of features: ', list(df1))

print('-------------------')
print('full data loaded')
print('-------------------')
```


```python
# для простоты возьмем 200 текстов
# проводим лематизацию этого датасета
y_data1 = []
# how we arrange cycle for lematization of 200 documents
for i in range(500):
    # getting doc
    text = str(df1['Text'][i])
    # remove quotes from text 
    s = text.replace('"', '')
    # how we are doing lematization
    s = m.lemmatize(s)
    s = ''.join(s)
    lemans=str(s)
    lemans = regex.sub('[!@#$1234567890#—ツ►๑۩۞۩•*”˜˜”*°°*`]', '', lemans)
    y_data1.append(lemans)    
    print('doc number : ', i)


```


```python
# теперь проводим векторизацию используя уже обученный векторизатор
count_vector=vec.transform(y_data1)

# transform doc to matrix
data1 = vec.transform(y_data1).toarray()
print('процесс векторизации закончен')
print('размер новой матрицы: ', data1.shape)
data_new = my_fit.transform(data1)
print(data_new.shape)


```


```python
# теперь можно сделать предсказание
# вот теперь подставить полученную матрицу data1 в уже обученный классификатор
y_predicted1 = model.predict(data_new)
# предсказание сентимента новых текстов.
print(y_predicted1)
```
