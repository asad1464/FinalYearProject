from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages
from pyforest import sns, plt
from user import models
from user.forms import *
from user.models import *
from django.http import FileResponse


def userlogin(request):
    return render(request,'user/userlogin.html')

def userpage(request):
    return render(request,'user/userpage.html')

def userregister(request):
    if request.method=='POST':
        form1=userForm(request.POST)
        if form1.is_valid():
            form1.save()
            print("succesfully saved the data")
            return render(request, "user/userlogin.html")
            #return HttpResponse("registreration succesfully completed")
        else:
            print("Please give other username")
            return HttpResponse("Please give other username")
    else:
        form=userForm()
        return render(request,"user/userregister.html",{"form":form})


def userlogincheck(request):
    if request.method == 'POST':
        mail = request.POST.get('mail')
        print(mail)
        spasswd = request.POST.get('spasswd')
        print(spasswd)
        try:
            check = usermodel.objects.get(email=mail, passwd=spasswd)
            # print('usid',usid,'pswd',pswd)
            print(check)
            request.session['name'] = check.name
            print("name",check.name)
            status = check.status
            print('status',status)
            if status == "Activated":
                request.session['email'] = check.email
                return render(request, 'user/userpage.html')
            else:
                messages.success(request, 'user  is not activated')
                return render(request, 'user/userlogin.html')
        except Exception as e:
            print('Exception is ',str(e))

        messages.success(request,'Invalid name and password')
        return render(request,'user/userlogin.html')

def adddata(request):
    if request.method=='POST':
        tx_price = request.POST.get["tx_price"]
        beds = request.POST.get["beds"]
        baths = request.POST.get["baths"]
        sqft = request.POST.get["sqft"]
        year_built = request.POST.get["year_built"]
        lot_size = request.POST.get["lot_size"]
        # property_type = request.POST.get["property_type"]
        # exterior_walls = request.POST.get["exterior_walls"]
        # roof = request.POST.get["roof"]
        basement = request.POST.get["basement"]
        restaurants = request.POST.get["restaurants"]
        groceries = request.POST.get["groceries"]
        nightlife = request.POST.get["nightlife"]
        cafes = request.POST.get["cafes"]
        shopping = request.POST.get["shopping"]
        arts_entertainment = request.POST.get["arts_entertainment"]
        beauty_spas = request.POST.get["beauty_spas"]
        active_life = request.POST.get["active_life"]
        median_age = request.POST.get["median_age"]
        married = request.POST.get["married"]
        college_grad = request.POST.get["college_grad"]
        property_tax = request.POST.get["property_tax"]
        insurance = request.POST.get["insurance"]
        median_school = request.POST.get["median_school"]
        num_school = request.POST.get["num_school"]
        tx_year = request.POST.get["tx_year"]

        csvdatamodel(tx_price=tx_price, beds=beds, baths=baths, sqft=sqft,
                                    year_built=year_built, lot_size=lot_size, basement=basement, restaurants=restaurants, groceries=groceries,
                                    nightlife=nightlife,
                                    cafes=cafes, shopping=shopping, arts_entertainment=arts_entertainment,
                                    beauty_spas=beauty_spas,
                                    active_life=active_life, median_age=median_age, married=married,
                                    college_grad=college_grad,
                                    property_tax=property_tax, insurance=insurance, median_school=median_school,
                                    num_school=num_school,
                                    tx_year=tx_year).save()
        return render(request,'user/adddata.html')

        #     return render(request, "user/adddata.html")
        #     #return HttpResponse("registreration succesfully completed")
    else:
        form=csvdatamodelForm()
        return render(request,"user/adddata.html",{"form":form})

def houseprediction(request):
    import numpy as np
    import pandas as pd
    import pyforest

    df = pd.read_csv('cleaned_df.csv')
    df.head()
    df.info()
    # print(df.shape)
    # print(df.isnull().sum())
    sns.heatmap(df.isnull())
    # plt.show()
    print(df.describe())
    plt.figure(figsize=(10, 8))
    sns.distplot(df['groceries'], color='g')
    # plt.show()
    # print("house median age-min:",df['housing_median_age'].min())
    # print("house median age-min:",df['housing_median_age'].max())

    # corr between feartures
    corr_matrix = df.corr()
    corr_df = corr_matrix['tx_price'].sort_values(ascending=False)
    print(corr_df)
    plt.figure(figsize=(12, 7))
    sns.heatmap(corr_matrix, annot=True)
    # plt.show()
    from pandas.plotting import scatter_matrix
    attr = ['tx_price', 'sqft', 'beds', 'groceries']
    scatter_matrix(df[attr], figsize=(16, 8), color='g', alpha=0.3)
    # plt.show()
    plt.figure(figsize=(16, 8))
    sns.pairplot(df[attr])
    # plt.show()
    df.plot(kind='scatter', x='beds', y='tx_price', c='g', figsize=(10, 7))
    plt.show()
    # handle categorical variable


    # fill null values
    # from sklearn.preprocessing import Imputer
    from sklearn.impute import SimpleImputer
    train_ft = df.drop(['tx_price'], axis=1)
    imputer = SimpleImputer(strategy='median')
    imputer.fit(train_ft)
    train_ft.median().values
    x = imputer.transform(train_ft)
    train_new_set = pd.DataFrame(x, columns=train_ft.columns)
    train_new_set.head()
    train_new_set.isna().sum()
    train_new_set.head()
    train_new_set.shape
    train_new_set.info()
    X = train_new_set.values
    Y = df['tx_price']
    # split the data

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2020)
    x_train.shape
    y_test.shape
    x_test.shape

    # model linear regression

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    predictions = lr.predict(x_test[:10])
    print("predictions:", predictions)
    y_train[:10]
    data = {'predicted': predictions, 'Actual': y_test[:10].values, 'Diff': (predictions - y_test[:10].values)}
    error_df = pd.DataFrame(data=data)
    htmltable = error_df.to_html()
    return HttpResponse(htmltable)
    # print("error diff:", error_df)
    #
    # return render(request,"user/houseprediction.html",{"errordiff":error_df})



    # model evaluation
# def pricepredict(request):
#     return render(request,"user/pricepredictions.html")
def pricepredictions(request):
    if request.method=='POST':
        form=pricemodelForm(request.POST)
        if form.is_valid():
            form.save()
        return render(request, "user/pricepredictions.html")


    else:
        form=pricemodelForm()
        return render(request,"user/pricepredictions.html",{'form':form})



def price(request,id):
    import numpy as np
    import pandas as pd
    import pyforest
    from user.models import pricemodel


    df = pd.read_csv('cleaned_data.csv')
    df.head()
    df.info()
    # print(df.shape)
    # print(df.isnull().sum())
    sns.heatmap(df.isnull())
    # plt.show()
    print(df.describe())

    from sklearn.impute import SimpleImputer
    train_ft = df.drop(['tx_price'], axis=1)
    imputer = SimpleImputer(strategy='median')
    imputer.fit(train_ft)

    x = imputer.transform(train_ft)
    train_new_set = pd.DataFrame(x, columns=train_ft.columns)
    train_new_set.head()
    train_new_set.isna().sum()
    train_new_set.head()
    train_new_set.shape
    train_new_set.info()
    X = train_new_set.values
    Y = df['tx_price']
    # split the data

    from sklearn.model_selection import train_test_split
    user_data = pricemodel.objects.get(id=id)
    usedata={
        # 'tx_price' : user_data.tx_price,
        # 'beds' : user_data.beds,
        # 'baths' : user_data.baths,
        'sqft' : user_data.sqft,
        # 'year_built' : user_data.year_built,
        # 'lot_size' : user_data.lot_size,
        # 'basement' : user_data.basement,
        'restaurants' : user_data.restaurants,
        'groceries' : user_data.groceries,
        'nightlife' : user_data.nightlife,
        'cafes' : user_data.cafes,
        'shopping' : user_data.shopping,
        'arts_entertainment' : user_data.arts_entertainment,
        'beauty_spas' : user_data.beauty_spas,
        'active_life' : user_data.active_life,
        'median_age' : user_data.median_age,
        'married' : user_data.married,
        'college_grad' : user_data.college_grad,
        'property_tax' : user_data.property_tax,
        # 'insurance' : user_data.insurance,
        'median_school' : user_data.median_school,
        'num_school' : user_data.num_school,
        'tx_year' : user_data.tx_year
    }
    userdata = pd.DataFrame(usedata, index=[0])
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    print(x_train.shape)



    # model linear regression

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()

    lr.fit(x_train, y_train)
    predictions = lr.predict(userdata)
    print(x_test)
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)
    accuracy = regressor.score(x_test, y_test)
    accuracy1= accuracy * 100
    print(accuracy * 100, '%')

    print("predictions:", predictions)

    import json
    data = {'predicted': predictions, 'accuracy1': accuracy1}
    error_df = pd.DataFrame(data=data)
    # json_records = error_df.reset_index().to_json(orient='records')
    # data = []
    # data = json.loads(json_records)
    # context = {'d': data}

    htmltable = error_df.to_html()
    return HttpResponse(htmltable)


    # return render(request, "user/userpricepridiction.html", {"errordiff": context})

def inputdisplay(request):
    inputvalues=pricemodel.objects.all()
    return render(request, "user/priceinput.html", {'inputvalue' : inputvalues})


def analysisreport(request):
    import numpy as np
    import pandas as pd
    import pyforest

    df = pd.read_csv('cleaned_data.csv')
    df.head()
    df.info()
    # print(df.shape)
    # print(df.isnull().sum())
    sns.heatmap(df.isnull())
    # plt.show()
    print(df.describe())
    plt.figure(figsize=(10, 8))
    sns.distplot(df['groceries'], color='g')
    # plt.show()
    # print("house median age-min:",df['housing_median_age'].min())
    # print("house median age-min:",df['housing_median_age'].max())

    # corr between feartures
    corr_matrix = df.corr()
    corr_df = corr_matrix['tx_price'].sort_values(ascending=False)
    print(corr_df)
    plt.figure(figsize=(12, 7))
    sns.heatmap(corr_matrix, annot=True)
    # plt.show()
    from pandas.plotting import scatter_matrix
    attr = ['tx_price', 'sqft', 'active_life', 'groceries']
    scatter_matrix(df[attr], figsize=(16, 8), color='g', alpha=0.3)
    # plt.show()
    plt.figure(figsize=(16, 8))
    sns.pairplot(df[attr])
    # plt.show()
    df.plot(kind='scatter', x='num_schools', y='tx_price', c='g', figsize=(10, 7))
    plt.show()
    from sklearn.impute import SimpleImputer
    train_ft = df.drop(['tx_price'], axis=1)
    imputer = SimpleImputer(strategy='median')
    imputer.fit(train_ft)
    train_ft.median().values
    x = imputer.transform(train_ft)
    train_new_set = pd.DataFrame(x, columns=train_ft.columns)
    train_new_set.head()
    train_new_set.isna().sum()
    train_new_set.head()
    train_new_set.shape
    train_new_set.info()
    X = train_new_set.values
    Y = df['tx_price']
    # split the data

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2020)
    x_train.shape
    y_test.shape
    x_test.shape

    # model linear regression

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    predictions = lr.predict(x_test[:10])
    print("predictions:", predictions)
    y_train[:10]
    data = {'predicted': predictions, 'Actual': y_test[:10].values, 'Diff': (predictions - y_test[:10].values)}
    error_df = pd.DataFrame(data=data)
    htmltable = error_df.to_html()
    return HttpResponse(htmltable)
    # print("error diff:", error_df)
    #
    # return render(request, "user/realesatesite.html", {"errordiff": error_df})


