from twikit import Client, TooManyRequests
import time
from datetime import datetime
import csv
from configparser import ConfigParser
from random import randint

MINIMUM_TWEETS = 10
QUERY = 'chatgpt'


# Login credentials
config = ConfigParser()
config.read('config.ini')
username = config['X']['username']
email = config['X']['email']
password = config['X']['password']

#* authenticate to x.com
# use the login credentials
''' loging to the x.com once using credentials then use cookies'''

client = Client(language='en-US')
# client.login(auth_info_1=username, auth_info_2=email, password=password)
# client.save_cookies('cookies.json ')

# use cookies
client.load_cookies('cookies.json')

# get data
async def get_data():
    tweets = await client.search_tweet(QUERY, product='Top')
    tweet_count = 0
    for tweet in tweets:
        tweet_count += 1
        tweet_data = [tweet_count, tweet.user.name, tweet.text, tweet.created_at, tweet.retweet_count, tweet.favourite_count]
        print(tweet_data)
        break

get_data()