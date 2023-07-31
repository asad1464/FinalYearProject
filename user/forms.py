from django.core import validators
from django import forms
from user.models import *
from django.core.exceptions import ValidationError
class userForm(forms.ModelForm):
    name = forms.CharField(widget=forms.TextInput(), required=True, max_length=100,)
    passwd = forms.CharField(widget=forms.PasswordInput(), required=True, max_length=100)
    cwpasswd = forms.CharField(widget=forms.PasswordInput(), required=True, max_length=100)
    email = forms.CharField(widget=forms.TextInput(),required=True)
    mobileno= forms.CharField(widget=forms.TextInput(), required=True, max_length=10,validators=[validators.MaxLengthValidator(10),validators.MinLengthValidator(10)])
    status = forms.CharField(widget=forms.HiddenInput(), initial='waiting', max_length=100)

    def __str__(self):
        return self.email

    class Meta:
        model=usermodel
        fields=['name','passwd','cwpasswd','email','mobileno','status']

    def clean(self):
        cleaned_data=super(userForm, self).clean()
        passwd=cleaned_data.get('passwd')
        cwpasswd=cleaned_data.get('cwpasswd')
        if passwd != cwpasswd:
            raise forms.ValidationError('Passwords Do not Match !')
            # Add this to check if both passwords are matching or not

                    # Add this to check if the email already exists in your database or not

    def clean_email(self):
        name = self.cleaned_data.get('name')
        email = self.cleaned_data.get('email')
        if email and usermodel.objects.filter(email=email).exclude(name=name).count():
            raise forms.ValidationError('This email is already in use! Try another email.')
        return email

                        # Add this to check if the username already exists in your database or not


    def clean_username(self):
        name = self.cleaned_data.get('name')
        email = self.cleaned_data.get('email')
        if name and usermodel.objects.filter(name=name).exclude(email=email).count():
            raise forms.ValidationError('This username has already been taken!')
        return name

class csvdatamodelForm(forms.ModelForm):

    tx_price = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    beds = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    baths = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    sqft = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    year_built = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    lot_size = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    basement = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    restaurants = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    groceries = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    nightlife = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    cafes = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    shopping = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    arts_entertainment = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    beauty_spas = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    active_life = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    median_age = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    married = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    college_grad = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    property_tax = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    insurance = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    median_school = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    num_school = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    tx_year = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)


    def __str__(self):
        return self.tx_price

    class Meta:
        model=csvdatamodel
        fields="__all__"



class pricemodelForm(forms.ModelForm):

    sqft = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    restaurants = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    groceries = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    nightlife = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    cafes = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    shopping = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    arts_entertainment = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    beauty_spas = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    active_life = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    median_age = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    married = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    college_grad = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    property_tax = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    median_school = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    num_school = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)
    tx_year = forms.CharField(widget=forms.TextInput(), required=True, max_length=10)


    def __str__(self):
        return self.sqft

    class Meta:
        model=pricemodel
        fields="__all__"




