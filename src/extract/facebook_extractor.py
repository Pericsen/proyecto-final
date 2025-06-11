import requests
import json
import pandas as pd
import sys
from datetime import datetime


import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    with open('./credentials/credentials_redes.json') as f:
        config = json.load(f)
except FileNotFoundError:
    logger.error("Credentials file not found. Make sure 'credentials_redes.json' exists.")
    sys.exit(1)
except json.JSONDecodeError:
    logger.error("Invalid JSON in credentials file.")
    sys.exit(1)

START_DATE = '2024-01-01'  
END_DATE = datetime.now().strftime('%Y-%m-%d')  


start_timestamp = int(datetime.strptime(START_DATE, '%Y-%m-%d').timestamp())
end_timestamp = int(datetime.now().timestamp())

# Extract credentials
ACCESS_TOKEN = config.get("pages_access_token")
PAGE_ID = config.get("page_id")

if not ACCESS_TOKEN or not PAGE_ID:
    logger.error("Missing credentials! pages_access_token or page_id not found in config file.")
    sys.exit(1)


API_VERSION = 'v22.0'  
BASE_URL = f'https://graph.facebook.com/{API_VERSION}'

def paginate(url, params):
    """Paginate through Facebook API responses with better error handling."""
    items = []
    page_count = 0
    
    while url:
        try:
            response = requests.get(url, params=params)
            response.raise_for_status() 
            resp = response.json()
            
            if 'error' in resp:
                logger.error(f"Facebook API error: {resp['error'].get('message')}")
                break
                
            items.extend(resp.get('data', []))
            page_count += 1
            
            if page_count % 5 == 0:
                logger.info(f"Retrieved {len(items)} items so far...")
                
            url = resp.get('paging', {}).get('next')
            params = {} 
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            break
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON response")
            break
    
    return items

def check_token():
    """Verify if the access token is valid and has proper permissions."""
    url = f'{BASE_URL}/debug_token'
    params = {
        'input_token': ACCESS_TOKEN,
        'access_token': ACCESS_TOKEN
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'data' in data:
            token_data = data['data']
            logger.info(f"Token valid: {token_data.get('is_valid', False)}")
            logger.info(f"Token expires: {token_data.get('expires_at') or 'Never'}")
            
            if 'scopes' in token_data:
                logger.info(f"Token permissions: {', '.join(token_data.get('scopes', []))}")
            
            return token_data.get('is_valid', False)
        return False
    except Exception as e:
        logger.error(f"Failed to check token: {e}")
        return False

def check_page_access():
    """Verify if we can access the page with the given token."""
    url = f'{BASE_URL}/{PAGE_ID}'
    params = {'access_token': ACCESS_TOKEN}
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'name' in data:
            logger.info(f"Successfully accessed page: {data['name']}")
            return True
        elif 'error' in data:
            logger.error(f"Page access error: {data['error'].get('message')}")
        return False
    except Exception as e:
        logger.error(f"Failed to check page access: {e}")
        return False

def get_page_posts(limit=200):
    """Get posts from the page with error handling."""
    logger.info(f"Fetching up to {limit} posts from page {PAGE_ID} (from {START_DATE} to {END_DATE})...")
    
    url = f'{BASE_URL}/{PAGE_ID}/posts'
    params = {
        'fields': ','.join([
            'id',
            'message',
            'created_time',
            'shares.summary(true)',
            'comments.summary(true).limit(0)',
            'reactions.summary(true).limit(0)'
        ]),
        'limit': limit,
        'since': start_timestamp,
        'until': end_timestamp,
        'access_token': ACCESS_TOKEN
    }
    
    posts = paginate(url, params)
    logger.info(f"Retrieved {len(posts)} posts")
    
    if posts and len(posts) > 0:
        first_post = posts[0]
        logger.info(f"Sample post ID: {first_post.get('id')}")
        logger.info(f"Has message: {'message' in first_post}")
        logger.info(f"Comment count: {first_post.get('comments', {}).get('summary', {}).get('total_count', 0)}")
    
    return posts

def get_comments_for_post(post_id, limit=200):
    """Get comments for a post with error handling."""
    logger.info(f"Fetching up to {limit} comments for post {post_id}...")
    
    url = f'{BASE_URL}/{post_id}/comments'
    params = {
        'fields': 'id,from,message,created_time,like_count',
        'limit': limit,
        'access_token': ACCESS_TOKEN
    }
    
    comments = paginate(url, params)
    logger.info(f"Retrieved {len(comments)} comments for post {post_id}")
    return comments

def build_comments_df(post_limit=200, comment_limit=200):
    """Build a DataFrame of comments with progress logging."""
    posts = get_page_posts(limit=post_limit)
    
    if not posts:
        logger.warning("No posts found. Check page ID and token permissions.")
        return pd.DataFrame()
    
    rows = []
    for i, p in enumerate(posts):
        logger.info(f"Processing post {i+1}/{len(posts)} (ID: {p['id']})")
        
        meta = {
            'post_id': p['id'],
            'post_message': p.get('message', ''),
            'post_time': p['created_time'],
            'post_shares': p.get('shares', {}).get('summary', {}).get('total_count', 0),
            'post_comments_count': p.get('comments', {}).get('summary', {}).get('total_count', 0),
            'post_reactions_count': p.get('reactions', {}).get('summary', {}).get('total_count', 0)
        }
        
        comment_count = meta['post_comments_count']
        if comment_count == 0:
            logger.info(f"Post has no comments, skipping comment retrieval")
            continue
            
        logger.info(f"Post has {comment_count} comments according to summary, fetching...")
        comments = get_comments_for_post(p['id'], limit=comment_limit)
        
        for c in comments:
            rows.append({
                **meta,
                'comment_id': c['id'],
                'comment_from': c.get('from', {}).get('name', ''),
                'comment_psid': c.get('from', {}).get('id', ''),
                'comment_text': c.get('message', ''),
                'comment_time': c['created_time'],
                'comment_likes': c.get('like_count', 0)
            })
    
    logger.info(f"Total comments collected: {len(rows)}")
    return pd.DataFrame(rows)

if __name__ == '__main__':
    # Run verification steps
    logger.info("=== Starting Facebook Comments Scraper ===")
    logger.info(f"Using API version: {API_VERSION}")
    
    # Check token validity
    logger.info("Checking token validity...")
    if not check_token():
        logger.warning("Token validation failed or returned warnings")
    
    # Check page access
    logger.info("Checking page access...")
    if not check_page_access():
        logger.error("Cannot access the page. Verify page ID and token permissions.")
        sys.exit(1)
    
    # Proceed with comment retrieval
    logger.info("Building comments DataFrame...")
    df_fb = build_comments_df(post_limit=100, comment_limit=100)
    
    if len(df_fb) == 0:
        logger.warning("No comments were retrieved!")
        # Save posts summary for analysis
        logger.info("Saving posts summary for debugging...")
        posts = get_page_posts(limit=50)
        if posts:
            posts_df = pd.DataFrame([{
                'post_id': p['id'],
                'post_message': p.get('message', '')[:50] + '...' if p.get('message') else '',
                'post_time': p['created_time'],
                'comment_count': p.get('comments', {}).get('summary', {}).get('total_count', 0)
            } for p in posts])
            posts_df.to_csv('fb_posts_debug.csv', index=False)
            logger.info(f"Saved {len(posts_df)} posts to fb_posts_debug.csv")
    else:
        # Save comments to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'fb_page_comments_{timestamp}.csv'
        df_fb.to_csv(filename, index=False)
        logger.info(f"Saved {len(df_fb)} comments to {filename}")
    
    logger.info("=== Script completed ===")