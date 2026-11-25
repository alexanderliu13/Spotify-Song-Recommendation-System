import re
import requests
import base64
import pandas as pd
import spotipy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def content_based_recommendations(input_track_id, music_df, music_features_scaled, num_recommendations=5):
    if input_track_id not in music_df['Track ID'].values:
        print(f"Sorry, track {input_track_id} is not in the database. Please choose a song that is.")
        return
    song_index = music_df[music_df['Track ID'] == input_track_id].index[0]
    similarity_scores = cosine_similarity([music_features_scaled[song_index]], music_features_scaled)
    scores_descending_sort = np.flip(similarity_scores.argsort())
    top_scores = scores_descending_sort[0][1:num_recommendations + 1]
    recommendations = music_df.iloc[top_scores][
        ['Song Name', 'Artists', 'Album Name', 'Release Date', 'Popularity']]
    for index, row in music_df.iterrows():
        music_df.loc[index, 'Similarity Score'] = round(similarity_scores[0][index] * 100, 2)
    return music_df


def hybrid_recommendations(input_track_id, music_df, music_features_scaled, num_recommendations=5,
                           target_popularity=50, popularity_weight=0.5):
    if input_track_id not in music_df['Track ID'].values:
        print(f"Sorry, track {input_track_id} is not in the database. Please choose a song that is.")
        return
    hybrid_recommendations = content_based_recommendations(input_track_id, music_df,
                                                           music_features_scaled, num_recommendations)
    popularity_match = 100 - (hybrid_recommendations['Popularity'] - target_popularity).abs()
    hybrid_recommendations['Recommendation Score'] = (popularity_weight * popularity_match +
                                                      (1 - popularity_weight) * hybrid_recommendations['Similarity Score'])
    hybrid_recommendations_sorted = hybrid_recommendations.sort_values(by='Recommendation Score', ascending=False)
    input_song = music_df.loc[music_df['Track ID'] == input_track_id].iloc[0]
    same_song = ((hybrid_recommendations_sorted['Song Name'] == input_song['Song Name']) &
                 (hybrid_recommendations_sorted['Artists'] == input_song['Artists']))
    hybrid_recommendations_sorted = hybrid_recommendations_sorted.loc[~same_song]
    top_hrs = hybrid_recommendations_sorted.head(num_recommendations)
    return top_hrs


def song_labels(music_df):
    labels = music_df['Song Name'] + ' — ' + music_df['Artists']
    duplicated = labels.duplicated(keep=False)
    labels[duplicated] = labels[duplicated] + ' (' + music_df.loc[duplicated, 'Album Name'] + ')'
    return dict(zip(music_df['Track ID'], labels))


def parse_playlist_id(playlist_link):
    match = re.fullmatch(r'(?:(?:https?://)?open\.spotify\.com/(?:intl-[\w-]+/)?(?:user/[^/?#]+/)?playlist/'
                         r'|spotify:(?:user:[^:]+:)?playlist:)?'
                         r'([A-Za-z0-9]{22})(?:[/?#].*)?', playlist_link.strip())
    return match.group(1) if match else None


def get_access_token(CLIENT_ID, CLIENT_SECRET):
    credentials = CLIENT_ID + ":" + CLIENT_SECRET
    credentials_base64 = base64.b64encode(credentials.encode())
    api_endpoint = 'https://accounts.spotify.com/api/token'
    headers = {
         'Authorization': f'Basic {credentials_base64.decode()}'
    }
    data = {
         'grant_type': 'client_credentials'
    }
    response = requests.post(api_endpoint, data=data, headers=headers)
    if response.status_code == 200:
        result = response.json()
        access_token = result['access_token']
        return access_token
    else:
        print("Error obtaining access token.")
        exit()


RECCOBEATS_AUDIO_FEATURES_URL = 'https://api.reccobeats.com/v1/audio-features'
RECCOBEATS_MAX_IDS = 40
AUDIO_FEATURES = {
    'danceability': 'Danceability', 'energy': 'Energy', 'key': 'Key', 'loudness': 'Loudness', 'mode': 'Mode',
    'speechiness': 'Speechiness', 'acousticness': 'Acousticness', 'instrumentalness': 'Instrumentalness',
    'liveness': 'Liveness', 'valence': 'Valence', 'tempo': 'Tempo',
}
PLAYLIST_COLUMNS = ['Song Name', 'Artists', 'Album Name', 'Album ID', 'Track ID', 'Popularity', 'Release Date',
                    'Duration (ms)', 'Explicit', 'External URLs', *AUDIO_FEATURES.values()]


def get_audio_features(track_ids):
    features = {}
    for i in range(0, len(track_ids), RECCOBEATS_MAX_IDS):
        batch = track_ids[i:i + RECCOBEATS_MAX_IDS]
        response = requests.get(RECCOBEATS_AUDIO_FEATURES_URL, params={'ids': ','.join(batch)}, timeout=30)
        response.raise_for_status()
        for item in response.json().get('content', []):
            track_id = item['href'].rsplit('/', 1)[-1]
            features[track_id] = {column: item.get(key) for key, column in AUDIO_FEATURES.items()}
    return features


def get_playlist_data(playlist_id, access_token):
    sp = spotipy.Spotify(auth=access_token)
    fields = ('next,items(track(id,name,artists(name),album(id,name,release_date),'
              'popularity,explicit,duration_ms,external_urls))')
    page = sp.playlist_items(playlist_id, fields=fields, additional_types=('track',))
    tracks = []
    while page:
        tracks += [item['track'] for item in page['items'] if item.get('track') and item['track'].get('id')]
        page = sp.next(page) if page.get('next') else None

    track_ids = list(dict.fromkeys(track['id'] for track in tracks))
    audio_features = get_audio_features(track_ids)

    music_data = []
    for song in tracks:
        song_features = audio_features.get(song['id'])
        if song_features is None:
            continue
        music_data.append({
            'Song Name': song['name'],
            'Artists': ', '.join(artist['name'] for artist in song['artists']),
            'Album Name': song['album']['name'],
            'Album ID': song['album']['id'],
            'Track ID': song['id'],
            'Popularity': song['popularity'],
            'Release Date': song['album']['release_date'],
            'Duration (ms)': song['duration_ms'],
            'Explicit': song['explicit'],
            'External URLs': song['external_urls'].get('spotify'),
            **song_features,
        })
    df = pd.DataFrame(music_data, columns=PLAYLIST_COLUMNS).drop_duplicates(subset='Track ID')
    df = df.dropna(subset=list(AUDIO_FEATURES.values())).reset_index(drop=True)
    df.attrs['total_tracks'] = len(track_ids)
    return df


def get_playlist_name(playlist_id, access_token):
    sp = spotipy.Spotify(auth=access_token)
    response = sp.playlist(playlist_id, fields = 'name')
    return response['name']


def get_playlist_image(playlist_id, access_token):
    sp = spotipy.Spotify(auth=access_token)
    response = sp.playlist(playlist_id, fields = 'images')
    return response['images'][0]['url']


def get_song_image(album_id, access_token):
    sp = spotipy.Spotify(auth=access_token)
    response = sp.album(album_id)
    return response
