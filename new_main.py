import asyncio
import tracemalloc
from twikit import Client, TooManyRequests
import time
from datetime import datetime
import csv
from configparser import ConfigParser
from random import randint
import os

MINIMUM_TWEETS = 20
QUERY = 'chatgpt'

# Login credentials
config = ConfigParser()
config.read('config.ini')
username = config['X']['username']
email = config['X']['email']
password = config['X']['password']

async def authenticate():
    client = Client(language='en-US')
    
    if os.path.exists('cookies.json'):
        client.load_cookies('cookies.json')
        print("Loaded existing cookies")
    else:
        try:
            await client.login(auth_info_1=username, auth_info_2=email, password=password)
            client.save_cookies('cookies.json')
            print("Logged in and saved new cookies")
        except Exception as e:
            print(f"Login failed: {e}")
            return None

    print("Authentication successful")
    return client

async def get_data(client):
    tweet_count = 0
    tweets = await client.search_tweet(QUERY, product='Top')

    with open('tweets.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Tweet Count', 'Username', 'Text', 'Created At', 'Retweets', 'Likes'])

        while tweet_count < MINIMUM_TWEETS:
            if tweets is None:
                print(f'{datetime.now()} - Getting tweets...')
                tweets = await client.search_tweet(QUERY, product='Top')
            else:
                print(f'{datetime.now()} - Getting next tweets...')
                tweets = await tweets.next()
            
            if not tweets:
                print(f'{datetime.now()} - No more tweets found')
                break

            for tweet in tweets:
                tweet_count += 1
                tweet_data = [tweet_count, tweet.user.name, tweet.text, tweet.created_at, tweet.retweet_count, tweet.favorite_count]
                writer.writerow(tweet_data)

    print(f'{datetime.now()} - Got {tweet_count} tweets')
    return tweet_count

async def main():
    tracemalloc.start()

    client = await authenticate()

    if client:
        tweet_count = await get_data(client)
        print(f"Total tweets scraped: {tweet_count}")
    else:
        print("Failed to authenticate")

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("\nTop 10 memory allocations:")
    for stat in top_stats[:10]:
        print(stat)

if __name__ == "__main__":
    asyncio.run(main())