import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

# trainSet.csv dosyasını okuyoruz ve ? içeren sütunları train setimizden kaldırdıktan sonra kayan indexi resetliyoruz.
data = pd.read_csv('trainSet.csv', na_values="?")
data  = data.dropna()
data = data.reset_index(drop=True)

# Aynı işlemi testSet.csv'ye de uyguluyoruz.
test_data = pd.read_csv('testSet.csv', na_values="?")
test_data  = test_data.dropna()
test_data = test_data.reset_index(drop=True)


# Train datamızdaki nümerik kolonlarımızı geçici bir dataframe üzerinde normalize ediyoruz ve bu sütunları kendi tablomuzdaki sütunlara eşitliyoruz. 
temp_data = data.select_dtypes(exclude='object')
columns = ['credit_amount', 'age']
min_max = preprocessing.MinMaxScaler()
temp_data[columns] = min_max.fit_transform(temp_data[columns]) 
data['credit_amount'] = temp_data['credit_amount']       ## Böylelikle elimizdeki tablonun nümerik kolonlarını minMaxScaler ile 0-1 arasında normalize etmiş oluyoruz.
data['age'] = temp_data['age']


# Aynı şekilde test datamızdaki nümerik kolonlara da normalizasyon uyguluyoruz.
test_temp_data = data.select_dtypes(exclude='object')
columns = ['credit_amount', 'age']
min_max = preprocessing.MinMaxScaler()
test_temp_data[columns] = min_max.fit_transform(test_temp_data[columns])
test_data['credit_amount'] = test_temp_data['credit_amount']
test_data['age'] = test_temp_data['age']


# Test ve train datamızda good -> positive, bad -> negative olduğundan dolayı; good için 1, bad için 0 değerlerini uyguluyoruz.
# Beklenen değer sütunu olan 'class' sütununu y dizisine atarak guassian naive-bayes için bir vektör elde ediyoruz.
# Test datasında ayırdığımız test_y dizisindeki değerleri daha sonra tp, tn, fp, fn değerlerini bulmak için kullanacağız

count = 0 
y = []
for value in data['class'].values:
    if value == 'good':
        data['class'].values[count] = 1
        y.append(data['class'].values[count])
        count += 1 
    else:     
        data['class'].values[count] = 0
        y.append(data['class'].values[count])
        count +=1

count = 0 
test_y = []
for value in test_data['class'].values:
    if value == 'good':
        test_data['class'].values[count] = 1
        test_y.append(test_data['class'].values[count])
        count += 1 
    else:     
        test_data['class'].values[count] = 0
        test_y.append(test_data['class'].values[count])
        count +=1

# Datamızdan prediction için beklenen değer sütunu olan 'class' column'ı kaldırıyoruz.
test_data.drop('class', inplace=True, axis=1) 
data.drop('class', inplace=True, axis=1)

# Train ve test datamızda kategorik sütunlarımızı nümerik değerlere dönüştürüyoruz.
data_n = data.astype('category')
data_n['credit_history'] = data_n['credit_history'].cat.codes
data_n['employment'] = data_n['employment'].cat.codes
data_n['property_magnitude'] = data_n['property_magnitude'].cat.codes

test_n = test_data.astype('category')
test_n['credit_history'] = test_n['credit_history'].cat.codes
test_n['employment'] = test_n['employment'].cat.codes
test_n['property_magnitude'] = test_n['property_magnitude'].cat.codes


# Elde ettiğimiz nümerik tablomuz ve biraz önce elde ettiğimiz y vektörümüzü train için, nümerik olan test_n tablomuzu test için kullanıyoruz. 
gnb = GaussianNB()
gnb.fit(data_n, y)
y_pred = gnb.predict(test_n)


# Test datamızdan ayırdığımız 'class' sütununu içeren y vektöründeki değerleri kullanarak tp, tn, fp, fn sayılarımızı buluyoruz.
tp = 0
tn = 0
fp = 0
fn = 0
count = 0
for pred in y_pred:
    if(y_pred[count] == 1 and test_y[count] == 1):
         tp += 1
    elif(y_pred[count] == 0 and test_y[count] == 0):
        tn += 1
    elif(y_pred[count] == 1 and  0 == test_y[count]):
          fp += 1
    elif(y_pred[count] == 0 and 1 == test_y[count]):
         fn += 1
    count+=1


# Elde ettiğimiz değerleri kullanarak gerekli formülleri tanımlayıp ekrana yazdırıyoruz.

accuracy = (tp + tn) / (tp + tn + fp + fn)
truePositiveRate = tp / (tp + fn)
trueNegativeRate = tn / (tn + fp)
falsePositiveRate = fp / (fp + tn) 
falseNegativeRate = fn / (fn + tp)

print("TP: ", tp, '\n', "TN: ", tn, '\n', "FP:", fp, '\n', "FN: ", fn, '\n',
"Accuracy : ", round(accuracy,3), '\n', "True Positive Rate: ", round(truePositiveRate,3) , '\n', "True Negative Rate: ", round(trueNegativeRate,3), '\n', 
"False Positive Rate: ", round(falsePositiveRate,3), '\n', "False Negative Rate", round(falseNegativeRate,3))
