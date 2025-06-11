import requests
from datetime import datetime
import json
import pandas as pd
import os

with open('./credentials/credentials_redes.json') as f:
    config = json.load(f)

ACCESS_TOKEN = config["access_token"]
USER_ID = config["ig_user_id"]

def get_recent_posts(limit):
    url = f"https://graph.instagram.com/v22.0/{USER_ID}/media"
    
    params = {
        "fields": "id, caption, timestamp, alt_text, comments_count, like_count, media_type, media_url",
        "limit": limit,
        "access_token": ACCESS_TOKEN
    }

    res = requests.get(url, params=params)
    return res.json().get("data", [])

def get_comments(media_id):
    comments = []

    url = f"https://graph.instagram.com/v22.0/{media_id}/comments"

    params = {
        "fields": "from, media, replies, like_count, text, timestamp",
        "access_token": ACCESS_TOKEN
    }

    while url:
        res = requests.get(url, params=params)
        data = res.json().get("data", [])
        comments.extend(data)
        url = res.json().get("paging", {}).get("next")
        params = {}

    return comments


def comments_to_csv(comments_dict, posts_list):
    
    posts_map = {post["id"]: post for post in posts_list}
    
    flat_data = []

    for post_id, comment_list in comments_dict.items():
        post = posts_map.get(post_id, {})
        for comment in comment_list:
            flat_data.append({
                "post_id": post_id,
                "comment_id": comment.get("id"),
                "username": comment.get("username"),
                "comment_text": comment.get("text"),
                "like_count": comment.get("like_count"),
                "timestamp": comment.get("timestamp"),
                "post_caption":        post.get("caption"),
                "post_time":           post.get("timestamp"),
                "post_alt_text":       post.get("alt_text"),
                "post_comments_count": post.get("comments_count"),
                "post_likes":          post.get("like_count"),
                "post_media_type":     post.get("media_type"),
                "post_media_url":      post.get("media_url"),
            })
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'instagram_comments_{timestamp}.csv'
    filepath = '../../data/raw/instagram'
    df = pd.DataFrame(flat_data)
    df.to_csv(os.path.join(filepath, filename), index=False)

def save_comments_to_csv(posts):
    all_comments = {}

    for post in posts:
        media_id = post["id"]
        print(f"Descargando comentarios de {media_id}")
        comments = get_comments(media_id)
        all_comments[media_id] = comments
    comments_to_csv(all_comments, posts)
    print("Comentarios guardados en comments.csv")


def main():
    posts = get_recent_posts(400)
    save_comments_to_csv(posts)

if __name__ == '__main__':
    main()