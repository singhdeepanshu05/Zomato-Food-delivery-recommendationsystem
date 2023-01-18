from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas  as pd
import json
df=pd.read_csv(r"C:\Users\91940\Downloads\NEW_FINAL.csv")
df_json=json.loads(df.to_json(orient='records'))
df2=pd.read_csv(r"C:\Users\91940\Downloads\NEW_FINAL.csv")
df3=pd.read_csv(r"C:\Users\91940\Downloads\NEW_FINAL.csv")

##sklearn for Cost For One
import sklearn
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df2["Cuisines"]=le.fit_transform(df2["Cuisines"])
from sklearn.preprocessing import LabelEncoder
le2=LabelEncoder()
df2["location"]=le2.fit_transform(df2["location"])
X2=df2.drop(['Unnamed: 0','Name','ID','rating', 'Dining_review','Delivery_Reviews','cost_for_one'],axis=1)
Y2=df2['cost_for_one']

# Sklearn Location Prediction
from sklearn.preprocessing import LabelEncoder
le3=LabelEncoder()
df3["Cuisines"]=le3.fit_transform(df3["Cuisines"])
X3=df3.drop(['Unnamed: 0','Name','ID','rating', 'Dining_review','Delivery_Reviews','location'],axis=1)
Y3=df3['location']

app=Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('first_site.html')
    #return 'Hello World'

@app.route('/predict',methods=['POST'])
def predict():

    Cuisine=request.form.get("Cuisine")
    Prefered_Location=request.form.get("Prelocation")
    Prefered_Price=request.form.get("Preprice")
    #all data is retrived

    #Average Cost For One
    Avg_cost=df[df["location"]==Prefered_Location]["cost_for_one"].mean()
    print(Prefered_Location)

    #Popular Cuisine
    m=pd.DataFrame(df[df["location"]==Prefered_Location]["Cuisines"].value_counts().rename_axis("Cuisines").reset_index(name="Count"))
    Popular_cuisine=m["Cuisines"][0]
    
    #Popular Restraurant
    Popular_restro=df[df["location"]==Prefered_Location].sort_values(by=["rating","Delivery_Reviews"],ascending=False)
    pop_rest_value=pd.DataFrame(Popular_restro["Name"]).reset_index(drop=True).iloc[0,0]

    #Serves
    ser=list(df[(df["location"]==Prefered_Location) & (df["Name"]==pop_rest_value)]["Cuisines"].unique())

    #Popular Restaurant that Serves your Cuisine
    Popular_restro_cui=df[(df["location"]==Prefered_Location) & (df["Cuisines"]==Cuisine)].sort_values(by=["rating","Delivery_Reviews"],ascending=False)
    pop_rest_value_cui=pd.DataFrame(Popular_restro_cui["Name"]).reset_index(drop=True).iloc[0,0]

    ##Ml Deployment

    # Cost_for_one
    from sklearn.tree import DecisionTreeClassifier
    dc2=DecisionTreeClassifier()
    dc2.fit(X2,Y2)
    case2=pd.DataFrame([{"location":Prefered_Location,"Cuisines":Cuisine}])
    case2["Cuisines"]=le.transform(case2["Cuisines"])
    case2["location"]=le2.transform(case2["location"])
    y_pred2=dc2.predict(case2)

    ##Prefered Loaction
    from sklearn.tree import DecisionTreeClassifier
    dc=DecisionTreeClassifier()
    dc.fit(X3,Y3)
    case1=pd.DataFrame([{"Cuisines":Cuisine,"cost_for_one":Prefered_Price}])
    case1["Cuisines"]=le3.transform(case1["Cuisines"])
    y_pred3=dc.predict(case1)


    return (render_template('first_site.html',Avg_price="{}".format(int(Avg_cost)),
    Re_price="{}".format(int(0.80*y_pred2[0])),
    Re_location="{}".format(y_pred3[0]),
    Pop_Cui="{}".format(Popular_cuisine),
    Pop_rest=pop_rest_value,
    serves=",".join(ser),
    pop_rest_cui=pop_rest_value_cui))



if __name__=="__main__":
    app.run(debug=True)