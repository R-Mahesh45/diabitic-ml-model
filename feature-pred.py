
import streamlit as st

st.title('You Bunch of Assholes:coffee:')

st.markdown('wave assholes! :kiss:')

st.markdown('Harshal gandu! :sweat_drops:')

st.markdown('Roshan Loduuuu! :dash:')

st.markdown('Kamlesh Virgin Mojito! :moyai: ')

st.markdown('Kunal Bhosdicha! :middle_finger:')

st.markdown('Amit Bhaioooo! :no_smoking:')

st.text('check your ass from diabetes')

st.markdown('i have messaged you the values enter that or check about your physical details and enter to know about yourself, this for a feature predctions from history to predict about your feature! :place_of_worship:')



# SVM Classification
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold





filename = 'pima.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]




dataframe


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3)




import matplotlib.pyplot as plt
plt.scatter(dataframe['plas'],dataframe['pedi'],c=dataframe['class'])
#try plot with other features as well mass and pedi



plt.scatter(dataframe['mass'],dataframe['skin'],c=dataframe['class'])



plt.scatter(dataframe['mass'],dataframe['pedi'],c=dataframe['class'])



plt.scatter(dataframe['age'],dataframe['pres'],c=dataframe['class'])



clf = SVC(kernel='rbf',gamma=0.0001)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
acc=accuracy_score(y_test,y_pred)*100
print("Accuracy= ",acc)
confusion_matrix(y_test,y_pred)




((134+39)/(134+39+40+18))*100




num_folds = 10
kfold = KFold(n_splits=num_folds)




# to get optimal value of gamma use grid search cv
clf = SVC()
# can put ['linear','rbf','poly'], give range for gamma i.e. C as a regularization parameter. Best out of it will be selected by algorithm. rbf-radial basis function
param_grid = [{'kernel':['rbf'],'gamma':[50,5,10,0.5,1,0.0001],'C':[1,15,14,13,12,11,10,0.1] }] # 6 X 8 =48 models will be created and will give best out of it.
gsv = GridSearchCV(clf,param_grid,cv=kfold)
gsv.fit(X_train,y_train)



gsv.best_params_ , gsv.best_score_ # 73% accuracy




clf = SVC(C=1,gamma=0.0001,kernel='rbf') # can change kernel and check accuracy
clf.fit(X_train , y_train) #build model
y_pred = clf.predict(X_test)#predict on test dataset
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy =", acc)
confusion_matrix(y_test, y_pred)



(134+37)/(134+37+49+11)


def predict(preg,plas,pres,skin,test,mass,pedi,age):
    prediction = model.predict([[preg,plas,pres,skin,test,mass,pedi,age]])
    return prediction
def main():
    st.title('Pima Indian Diabetese')
    
    preg = st.number_input('Pregnancy: ')
    plas = st.number_input('Plasma: ')
    pres = st.number_input('Pres: ')
    skin = st.number_input('Skin: ')
    test = st.number_input('Test: ')
    mass = st.number_input('Mass: ')
    pedi = st.number_input('pedigree: ')
    age = st.number_input('Age: ')
    
    
    if st.button('Predict'):
        result = predict(preg,plas,pres,skin,test,mass,pedi,age)
        if result==0:
            st.success("your ass has been saved from diabetic")
        else:
            st.success("you fucked up! you are having diabetis")
            
            
if __name__ == '__main__':
    main()        


















