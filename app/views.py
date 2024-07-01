from django.shortcuts import render, HttpResponse, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
# for anlyze
import hashlib
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import yfinance as yf

# Create your views here.


@login_required(login_url='login')
def HomePage(request):
    if request.method == "POST":
        name = request.POST.get("stock")
        start_date = request.POST.get("start")
        end_date = request.POST.get("end")
        username = request.user.username
        print(username)
        data = tenseflow_analyze(name, start_date, end_date, username)

        result = data[0]
        img_input = data[1]
        img_output = data[2]

        return render(request, "home.html", {'result': result, 'img_input': img_input, 'img_output': img_output})

    return render(request, "home.html")


def SignupPage(request):
    if request.method == "POST":
        uname = request.POST.get("username")
        email = request.POST.get("email")
        pass1 = request.POST.get("password1")
        pass2 = request.POST.get("password2")

        if pass1 != pass2:
            return HttpResponse("Password not confirmed!")
        else:
            my_user = User.objects.create_user(uname, email, pass1)
            my_user.save()
            return redirect('login')
            print(uname, email,  hashlib.sha1(pass1.encode("UTF-8")),
                  hashlib.sha1(pass2.encode("UTF-8")))
    return render(request, "signup.html")


def LoginPage(request):
    if request.method == "POST":
        username = request.POST.get('username')
        pass1 = request.POST.get('pass')
        user = authenticate(request, username=username, password=pass1)
        if user is not None:
            login(request, user)
            return redirect('home')
            print(username,  hashlib.sha1(pass1.encode("UTF-8")))
        else:
            return HttpResponse("Username or passwword incorrect!")

    return render(request, "login.html")


def LogoutPage(request):
    logout(request)
    return redirect('login')

# Analyze


def tenseflow_analyze(name, start_date, end_date, username):

    # Getting finance date with yfinance
    df = yf.download(name, start=start_date, end=end_date)

    # start work with data
    rows = df.shape
    print(rows)

    # Create dataframe
    data = df.filter(['Close'])
    dataset = data.values
    training_data_len = math.ceil(len(dataset) * .8)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Create data set
    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []

    for i in range(10, len(train_data)):
        x_train.append(train_data[i-10:i, 0])
        y_train.append(train_data[i, 0])
        if i <= 10:
            print(x_train)
            print(y_train)

    # Convert the x_train and y_train
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_train.shape

    # Build model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True,
              input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    # Create the testing data set
    test_data = scaled_data[training_data_len - 10:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(10, len(test_data)):
        x_test.append(test_data[i-10:i, 0])

    # Conver data
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Get the root mean squared error
    rmse = np.sqrt(np.mean(predictions - y_test)**2)
    print(rmse)

    # Plot the data
    train = data[:training_data_len]

    valid = data[training_data_len:]
    valid['Predictions'] = predictions

    # Show the valid and predicted prices
    # count = x_test.shape[0]
    date = str(start_date + ":" + end_date)
    print(valid)

    string = str(name + ": " + str(valid))

    data[0] = string
    data[1] = df['Close']
    data[2] = valid['Predictions']

    return (data)
