from django.db import models

class usermodel(models.Model):
    name = models.CharField(max_length=50)
    email = models.EmailField()
    passwd = models.CharField(max_length=40)
    cwpasswd = models.CharField(max_length=40)
    mobileno = models.CharField(max_length=12, default="", editable=True)
    status = models.CharField(max_length=40, default="", editable=True)

    def  __str__(self):
        return self.email

    class Meta:
        db_table='userregister'


class csvdatamodel(models.Model):
    # longitude = models.CharField(max_length=50)
    # latitude = models.EmailField()
    # housing_median_age = models.CharField(max_length=40)
    # total_rooms = models.CharField(max_length=40)
    # total_bedrooms = models.CharField(max_length=50, default="", editable=True)
    # population = models.CharField(max_length=40, default="", editable=True)
    # households = models.CharField(max_length=40, default="", editable=True)
    # median_income = models.CharField(max_length=40, default="", editable=True)
    # median_house_value = models.CharField(max_length=40, default="", editable=True)
    # ocean_proximity = models.CharField(max_length=40, default="", editable=True)
    tx_price = models.CharField(max_length=50, default="", editable=True)
    beds = models.CharField(max_length=50, default="", editable=True)
    baths = models.CharField(max_length=50, default="", editable=True)
    sqft = models.CharField(max_length=50)
    year_built = models.CharField(max_length=50)
    lot_size = models.CharField(max_length=50)
    # property_type = models.CharField(max_length=50, default="", editable=True)
    # exterior_walls = models.CharField(max_length=50)
    # roof = models.CharField(max_length=50)
    basement = models.CharField(max_length=50)
    restaurants = models.CharField(max_length=50)
    groceries = models.CharField(max_length=50)
    nightlife = models.CharField(max_length=50)
    cafes = models.CharField(max_length=50)
    shopping = models.CharField(max_length=50)
    arts_entertainment = models.CharField(max_length=50)
    beauty_spas = models.CharField(max_length=50)
    active_life = models.CharField(max_length=50)
    median_age = models.CharField(max_length=50)
    married = models.CharField(max_length=50)
    college_grad = models.CharField(max_length=50)
    property_tax = models.CharField(max_length=50, default="", editable=True)
    insurance = models.CharField(max_length=50,default="", editable=True)
    median_school = models.CharField(max_length=50)
    num_school = models.CharField(max_length=50)
    tx_year = models.CharField(max_length=50)

    class Meta:
        db_table='csvdatamodel'


class pricemodel(models.Model):

    # tx_price = models.CharField(max_length=50, default="", editable=True)
    # beds = models.CharField(max_length=50, default="", editable=True)
    # baths = models.CharField(max_length=50, default="", editable=True)
    sqft = models.CharField(max_length=50)
    # year_built = models.CharField(max_length=50)
    # lot_size = models.CharField(max_length=50)
    # property_type = models.CharField(max_length=50, default="", editable=True)
    # exterior_walls = models.CharField(max_length=50)
    # roof = models.CharField(max_length=50)
    # basement = models.CharField(max_length=50)
    restaurants = models.CharField(max_length=50)
    groceries = models.CharField(max_length=50)
    nightlife = models.CharField(max_length=50)
    cafes = models.CharField(max_length=50)
    shopping = models.CharField(max_length=50)
    arts_entertainment = models.CharField(max_length=50)
    beauty_spas = models.CharField(max_length=50)
    active_life = models.CharField(max_length=50)
    median_age = models.CharField(max_length=50)
    married = models.CharField(max_length=50)
    college_grad = models.CharField(max_length=50)
    property_tax = models.CharField(max_length=50, default="", editable=True)
    # insurance = models.CharField(max_length=50,default="", editable=True)
    median_school = models.CharField(max_length=50)
    num_school = models.CharField(max_length=50)
    tx_year = models.CharField(max_length=50)

    class Meta:
        db_table='pricemodel'







