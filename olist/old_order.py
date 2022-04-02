import pandas as pd
import numpy as np
import datetime as dt
from functools import reduce
from olist.utils import haversine_distance
from olist.data import Olist


class Order:
    '''
    DataFrames containing all orders as index,
    and various properties of these orders as columns
    '''
    def __init__(self):
        # Assign an attribute ".data" to all new instances of Order
        self.data = Olist().get_data()

    def get_wait_time(self, is_delivered=True):
        """
        Returns a DataFrame with:
        [order_id, wait_time, expected_wait_time, delay_vs_expected, order_status]
        and filters out non-delivered orders unless specified
        """
        # Hint: Within this instance method, you have access to the instance of the class Order in the variable self, as well as all its attributes
        orders = self.data['orders'][['order_id', 'order_status', 'order_purchase_timestamp', \
                            'order_delivered_customer_date', 'order_estimated_delivery_date']] \
                            .query("order_status == 'delivered'").copy()
        date_columns = ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']
        orders[date_columns] = orders[date_columns].apply(pd.to_datetime)
        orders['wait_time'] = (orders['order_delivered_customer_date'].dt.date - orders['order_purchase_timestamp'].dt.date) / dt.timedelta(days=1)
        orders['expected_wait_time'] = (orders['order_estimated_delivery_date'].dt.date - orders['order_purchase_timestamp'].dt.date) / dt.timedelta(days=1)
        orders['delay_vs_expected'] = ((orders['order_delivered_customer_date'].dt.date - orders['order_estimated_delivery_date'].dt.date) / dt.timedelta(days=1)).apply(lambda x: 0 if x <= 0 else x)
        column_order = ['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected', 'order_status']
        return orders[column_order]

    def get_review_score(self):
        """
        Returns a DataFrame with:
        order_id, dim_is_five_star, dim_is_one_star, review_score
        """
        reviews = self.data['order_reviews'][['order_id', 'review_score']].copy()
        reviews['dim_is_five_star'] = reviews['review_score'].apply(lambda x: 1 if x == 5 else 0)
        reviews['dim_is_one_star'] = reviews['review_score'].apply(lambda x: 1 if x == 1 else 0)
        column_order = ['order_id', 'dim_is_five_star', 'dim_is_one_star', 'review_score']
        return reviews.reindex(columns=column_order)

    def get_number_products(self):
        """
        Returns a DataFrame with:
        order_id, number_of_products
        """
        order_items = self.data['order_items'][['order_id', 'order_item_id']].copy()
        return order_items.groupby('order_id').order_item_id.count().reset_index().rename(columns={'order_item_id': 'number_of_products'})

    def get_number_sellers(self):
        """
        Returns a DataFrame with:
        order_id, number_of_sellers
        """
        order_items = self.data['order_items'][['order_id', 'seller_id', 'order_item_id']].copy()
        return order_items.groupby(['order_id']).seller_id.count().reset_index().rename(columns={'seller_id': 'number_of_sellers'})

    def get_price_and_freight(self):
        """
        Returns a DataFrame with:
        order_id, price, freight_value
        """
        order_items = self.data['order_items'][['order_id', 'price', 'freight_value']].copy()
        return order_items.groupby('order_id').agg({'price':'sum','freight_value':'sum'}).reset_index()

    # Optional
    def get_distance_seller_customer(self):
        """
        Returns a DataFrame with:
        order_id, distance_seller_customer
        """
         # import data
        data = self.data
        orders = data['orders']
        order_items = data['order_items']
        sellers = data['sellers']
        customers = data['customers']

        # Since one zip code can map to multiple (lat, lng), take the first one
        geo = data['geolocation']
        geo = geo.groupby('geolocation_zip_code_prefix',
                          as_index=False).first()

        # Merge geo_location for sellers
        sellers_mask_columns = [
            'seller_id', 'seller_zip_code_prefix', 'geolocation_lat', 'geolocation_lng'
        ]

        sellers_geo = sellers.merge(
            geo,
            how='left',
            left_on='seller_zip_code_prefix',
            right_on='geolocation_zip_code_prefix')[sellers_mask_columns]

        # Merge geo_location for customers
        customers_mask_columns = ['customer_id', 'customer_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']

        customers_geo = customers.merge(
            geo,
            how='left',
            left_on='customer_zip_code_prefix',
            right_on='geolocation_zip_code_prefix')[customers_mask_columns]

        # Match customers with sellers in one table
        customers_sellers = customers.merge(orders, on='customer_id')\
            .merge(order_items, on='order_id')\
            .merge(sellers, on='seller_id')\
            [['order_id', 'customer_id','customer_zip_code_prefix', 'seller_id', 'seller_zip_code_prefix']]

        # Add the geoloc
        matching_geo = customers_sellers.merge(sellers_geo,
                                            on='seller_id')\
            .merge(customers_geo,
                   on='customer_id',
                   suffixes=('_seller',
                             '_customer'))
        # Remove na()
        matching_geo = matching_geo.dropna()

        matching_geo.loc[:, 'distance_seller_customer'] =\
            matching_geo.apply(lambda row:
                               haversine_distance(row['geolocation_lng_seller'],
                                                  row['geolocation_lat_seller'],
                                                  row['geolocation_lng_customer'],
                                                  row['geolocation_lat_customer']),
                               axis=1)
        # Since an order can have multiple sellers,
        # return the average of the distance per order
        order_distance =\
            matching_geo.groupby('order_id',
                                 as_index=False).agg({'distance_seller_customer':
                                                      'mean'})

        return order_distance

    def get_training_data(self,
                          is_delivered=True,
                          with_distance_seller_customer=False):
        """
        Returns a clean DataFrame (without NaN), with the all following columns:
        ['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected',
        'order_status', 'dim_is_five_star', 'dim_is_one_star', 'review_score',
        'number_of_products', 'number_of_sellers', 'price', 'freight_value',
        'distance_seller_customer']
        """
        # Hint: make sure to re-use your instance methods defined above
        wait_time = self.get_wait_time()
        review_score = self.get_review_score()
        number_products = self.get_number_products()
        number_sellers = self.get_number_sellers()
        price_and_freight = self.get_price_and_freight()
        distance = self.get_distance_seller_customer()
        dfs = [wait_time, review_score, number_products, number_sellers, price_and_freight, distance]
        df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['order_id']), dfs)
        return df_merged.dropna()
