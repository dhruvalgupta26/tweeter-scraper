import json
from langchain_text_splitters import CharacterTextSplitter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import pandas as pd
from bs4 import BeautifulSoup
import os
import logging
from langchain.vectorstores import Pinecone as LangchainPinecone
from sqlalchemy import text
from models import load_config, setup_logs, create_sql_engine
from pinecone import Pinecone, ServerlessSpec
from uuid import uuid4
# from textwrap import shorten
from langchain_pinecone import PineconeVectorStore
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Load config with error handling
try:
    config = load_config()
    setup_logs(config)
except Exception as e:
    logging.error(f"Failed to load configuration or set up logs: {str(e)}")
    raise

def get_products_data(config):
    if config['database']['type'] == 'sql':
        engine = create_sql_engine(config)
        with engine.connect() as conn:
            field_mapping = config['field_mapping']['products']

            # Extract column and table names from the config
            stock_code_column = field_mapping['StockCode']['column_name']
            stock_code_table = field_mapping['StockCode']['table_name']

            description_column = field_mapping['Description']['column_name']
            description_table = field_mapping['Description']['table_name']

            unit_price_column = field_mapping['UnitPrice']['column_name']
            unit_price_table = field_mapping['UnitPrice']['table_name']

            # Construct the SQL query with JOINs if columns are from different tables
            if stock_code_table == description_table == unit_price_table:
                sql_query = text(f"""
                    SELECT {stock_code_column} AS StockCode,
                           {description_column} AS Description,
                           {unit_price_column} AS UnitPrice
                    FROM {stock_code_table}
                """)
                products_data = pd.read_sql(sql_query, conn)

            else:
                # query for StockCode
                stock_code_query = f"SELECT {stock_code_column} from {stock_code_table}"
                stock_code_series = pd.read_sql_query(stock_code_query,conn)[stock_code_column]

                # query for Description
                description_query = f"SELECT {description_column} from {description_table}"
                description_series = pd.read_sql_query(description_query,conn)[description_column]

                # query for UnitPrice
                unit_price_query = f"SELECT {unit_price_column} from {unit_price_table}"
                unit_price_series = pd.read_sql_query(unit_price_query,conn)[unit_price_column]

                # concat the series
                products_data = pd.concat([stock_code_series,description_series,unit_price_series],axis=1, keys=['StockCode, Description, UnitPrice'])
        return products_data

def clean_and_preprocess_data(df):
    """Clean the data."""
    try:
        # Clean HTML from description
        def clean_html(html_string):
            if pd.isna(html_string):
                return ""
            soup = BeautifulSoup(html_string, "html.parser")
            return soup.get_text(strip=True)

        df['Description'] = df['Description'].apply(clean_html)
        df = df.dropna(subset=['StockCode', 'Description','UnitPrice'])
        df = df.drop_duplicates(subset=['Description'])
        df = df[df['UnitPrice'] > 0]
        df = df[(df['Description'] != 'testing') & (df['Description'] != 'test')]
        return df
    except Exception as e:
        logging.error(f"Error cleaning data: {str(e)}")
        raise


# def load_or_create_vectorstore(config):
#     # save_dir = os.path.join(config['base_dir'], 'data', 'vectorstore')
    
#     try:
#         pinecone_api_key = config['vector_db']['pinecone_api_key']
#         pc = Pinecone(api_key=pinecone_api_key)
#         index_name = config['vector_db']['pinecone_index_name']
#         # Initialize Pinecone index
#         index = pc.Index(index_name)

#         if index:
#             # Load the existing vectorstore
#             embeddings = HuggingFaceEmbeddings(model_name='distiluse-base-multilingual-cased-v2')
#             vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
#             logging.info(f'Loaded existing vectorstore from pinecone db')
#         else:
#             # Create and save a new vectorstore
#             vectorstore = create_and_save_vectorstore(config)
#         return vectorstore
#     except Exception as e:
#         logging.error(f"Failed to load or create vectorstore: {str(e)}")
#         raise

def create_and_save_vectorstore(config):
    try:
        # Initialize Pinecone client
        pinecone_api_key = config['vector_db']['pinecone_api_key']
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Load and preprocess data
        df = get_products_data(config)
        df = clean_and_preprocess_data(df)
        df_dir = os.path.join(config['base_dir'], 'data', 'processed', 'products.csv')
        df.to_csv(df_dir, index=False)
        logging.info('Saved products data as a CSV file')
        
        # Combine features into a single string for each row
        df['combined_features'] = df.apply(lambda x: f"stockcode:{x['StockCode']}, description:{x['Description']}, price:{x['UnitPrice']}", axis=1)
        
        # Convert DataFrame rows to a list of documents
        documents = [
            Document(page_content=row['combined_features'], metadata={
                "stockcode": row['StockCode'],
                "description": row['Description'],
                "price": str(row['UnitPrice'])  # Convert to string to ensure consistent serialization
            })
            for _, row in df.iterrows()
        ]
        logging.info('Created documents')
        
        # Split the documents
        splitter = CharacterTextSplitter(chunk_size=900, chunk_overlap=20)
        splits = splitter.split_documents(documents)
        logging.info('Converted the documents into splits')
        
        # Embedding model
        embeddings = HuggingFaceEmbeddings(model_name='distiluse-base-multilingual-cased-v2')
        
        # Create Pinecone index if it doesn't exist
        index_name = str(config['vector_db']['pinecone_index_name'])
        if not index_name:
            raise ValueError("Pinecone index name is empty or not set")
        
        # Initialize Pinecone index
        index = pc.Index(index_name)
        
        # Function to check metadata size
        def is_valid_metadata_size(metadata, max_size=40960):
            return len(json.dumps(metadata).encode('utf-8')) <= max_size
        
        # # Function to truncate metadata if necessary
        # def truncate_metadata(metadata, max_size=40960):
        #     while len(json.dumps(metadata).encode('utf-8')) > max_size:
        #         if len(metadata['description']) > 10:
        #             metadata['description'] = metadata['description'][:30] + '...'
        #         else:
        #             # If description is already very short, we can't reduce further
        #             logging.warning(f"Unable to reduce metadata size below limit for: {metadata}")
        #             return None
        #     return metadata
        
        # Populate Pinecone vectorstore
        vectorstore = PineconeVectorStore(index=index, embedding=embeddings)
        
        valid_uuids = []
        valid_splits = []
        
        for split in splits:
            metadata = split.metadata
            
            # Check if the metadata size is within the limit
            if not is_valid_metadata_size(metadata):
                # metadata = truncate_metadata(metadata)
                if metadata is None:
                    continue  # Skip this split if we can't reduce metadata size
            
            valid_uuids.append(str(uuid4()))
            split.metadata = metadata  # Update the split with potentially truncated metadata
            valid_splits.append(split)
        
        # Add valid documents to the vector store
        if valid_splits:
            vectorstore.add_documents(documents=valid_splits, ids=valid_uuids)
            logging.info(f'Stored {len(valid_splits)} splits in Pinecone vectorstore')
        else:
            logging.warning('No valid splits to store in vectorstore')
        
        # Save the vectorstore
        save_dir = os.path.join(config['base_dir'], 'data', 'vectorstore')
        os.makedirs(save_dir, exist_ok=True)
        vectorstore.save_local(save_dir)
        logging.info(f'Saved vectorstore to {save_dir}')
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

# Assuming this function exists elsewhere in your code
def shorten(text, width, placeholder):
    if len(text) <= width:
        return text
    return text[:width-len(placeholder)] + placeholder


# # Load or create the vectorstore
# try:
#     # vectorstore = load_or_create_vectorstore(config)
#     vectorstore = create_and_save_vectorstore(config)
# except Exception as e:
#     logging.error(f"Failed to initialize vectorstore: {str(e)}")
#     raise

def perform_similarity_search(query, k=5):
    try:
        logging.info(f"Performing similarity search ")
        similar_docs = vectorstore.similarity_search_with_score(query, k=k)
        logging.info('Similarity search completed')

        # Format the results
        results = []
        for doc, score in similar_docs:
            results.append({
                "stockcode": doc.metadata["stockcode"],
                "description": doc.metadata["description"],
                "price": float(doc.metadata["price"]),  # Convert to float for JSON serialization
                "similarity_score": float(score)  # Convert to float for JSON serialization
            })

        return results
    except Exception as e:
        logging.error(f"Error during similarity search: {str(e)}")
        raise

def recommend_similar_products(stockcode, description, price):
    try:
        user_input = f"stockcode:{stockcode}, description:{description}, price:{price}"

        # Perform similarity search
        similar_products = perform_similarity_search(str(user_input))
        return similar_products
    except Exception as e:
        logging.error(f"Error during product recommendation: {str(e)}")
        raise

# Define the API
app = FastAPI()
# Load or create the vectorstore
try:
    vectorstore = create_and_save_vectorstore(config)
except Exception as e:
    logging.error(f"Failed to initialize vectorstore: {str(e)}")
    raise

def perform_similarity_search(query, k=5):
    try:
        logging.info(f"Performing similarity search ")
        similar_docs = vectorstore.similarity_search_with_score(query, k=k)
        logging.info('Similarity search completed')

        # Format the results
        results = []
        for doc, score in similar_docs:
            results.append({
                "stockcode": doc.metadata["stockcode"],
                "description": doc.metadata["description"],
                "price": float(doc.metadata["price"]),  # Convert to float for JSON serialization
                "similarity_score": float(score)  # Convert to float for JSON serialization
            })

        return results
    except Exception as e:
        logging.error(f"Error during similarity search: {str(e)}")
        raise

def recommend_similar_products(stockcode, description, price):
    try:
        user_input = f"stockcode:{stockcode}, description:{description}, price:{price}"

        # Perform similarity search
        similar_products = perform_similarity_search(str(user_input))
        return similar_products
    except Exception as e:
        logging.error(f"Error during product recommendation: {str(e)}")
        raise

# Define the API
app = FastAPI()

class ProductRequest(BaseModel):
    stockcode: str
    description: str
    price: float

@app.get("/personalised-recommend")
async def get_cb_recommendations(request: ProductRequest):
    try:
        stockcode = int(request.stockcode)
        description = request.description
        price = request.price
        # recommendations_json,
        similar_products = recommend_similar_products(stockcode, description, price)

        # Filter out the product with the same stock code
        filtered_products = [product for product in similar_products if product['stockcode'] != stockcode]

        response = {
            "similar_products": filtered_products
        }
        return response
    except Exception as e:
        logging.error(f"Failed to process request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="localhost", port=8080)
    except Exception as e:
        logging.error(f"Failed to start the server: {str(e)}")
