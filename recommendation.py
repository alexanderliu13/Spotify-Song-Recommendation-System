import requests
import base64
import pandas as pd
import spotipy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



def content_based_recommendations(input_song_name, music_df, music_features_scaled, num_recommendations=5):
    if input_song_name not in music_df['Song Name'].values:
        print(f"Sorry, {input_song_name} is not in the database. Please choose a song that is.")
        return
    song_index = music_df[music_df['Song Name'] == input_song_name].index[0]
    similarity_scores = cosine_similarity([music_features_scaled[song_index]], music_features_scaled)
    scores_descending_sort = np.flip(similarity_scores.argsort())
    top_scores = scores_descending_sort[0][1:num_recommendations + 1]
    recommendations = music_df.iloc[top_scores][
        ['Song Name', 'Artists', 'Album Name', 'Release Date', 'Popularity']]
    for index, row in music_df.iterrows():
        music_df.loc[index, 'Similarity Score'] = round(similarity_scores[0][index] * 100, 2)
    return music_df

def hybrid_recommendations(input_song_name, music_df, music_features_scaled, num_recommendations=5, alpha=0.5):
    if input_song_name not in music_df['Song Name'].values:
        print(f"Sorry, {input_song_name} is not in the database. Please choose a song that is.")
        return
    content_recommendation = content_based_recommendations(input_song_name, music_df,
                                                           music_features_scaled, num_recommendations)
    hybrid_recommendations = content_recommendation
    for index, row in hybrid_recommendations.iterrows():
        hybrid_recommendations.loc[index, 'Recommendation Score'] = (alpha *
                                                                     hybrid_recommendations.loc[index, 'Popularity'] +
                                                                     (1 - alpha) * hybrid_recommendations.loc[index, 'Similarity Score'])
    hybrid_recommendations_sorted = hybrid_recommendations.sort_values(by='Recommendation Score', ascending=False)
    hybrid_recommendations_sorted = hybrid_recommendations_sorted.loc[hybrid_recommendations_sorted["Song Name"] != input_song_name]
    top_hrs = hybrid_recommendations_sorted.head(num_recommendations)
    return top_hrs

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

def get_playlist_data(playlist_id, access_token):
    sp = spotipy.Spotify(auth=access_token)
    playlist_songs = sp.playlist_tracks(playlist_id, 'items(track(id, name, artists, album(id, name)))')
    music_data = []
    for info in playlist_songs['items']:
        song = info['track']
        song_name = song['name']
        artists = ', '.join([artist['name'] for artist in song['artists']])
        album_name = song['album']['name']
        album_id = song['album']['id']
        song_id = song['id']
        if song_id != 'Not available':
            try:
                audio_features = sp.audio_features(song_id)[0]
            except TypeError:
                audio_features = None
        else:
            audio_features = None
        if album_id != 'Not available':
            album_info = sp.album(album_id)
            release_date = album_info['release_date']
        else:
            album_info = None
            release_date = None
        if song_id != 'Not available':
            info = sp.track(song_id)
            popularity = info['popularity']

        else:
            info = None
            popularity = None

        song_data = {
            'Song Name': song_name,
            'Artists': artists,
            'Album Name': album_name,
            'Album ID': album_id,
            'Track ID': song_id,
            'Popularity': popularity,
            'Release Date': release_date,
            'Duration (ms)': audio_features['duration_ms'] if audio_features else None,
            'Explicit': info.get('explicit', None),
            'External URLs': info.get('external_urls', {}).get('spotify', None),
            'Danceability': audio_features['danceability'] if audio_features else None,
            'Energy': audio_features['energy'] if audio_features else None,
            'Key': audio_features['key'] if audio_features else None,
            'Loudness': audio_features['loudness'] if audio_features else None,
            'Mode': audio_features['mode'] if audio_features else None,
            'Speechiness': audio_features['speechiness'] if audio_features else None,
            'Acousticness': audio_features['acousticness'] if audio_features else None,
            'Instrumentalness': audio_features['instrumentalness'] if audio_features else None,
            'Liveness': audio_features['liveness'] if audio_features else None,
            'Valence': audio_features['valence'] if audio_features else None,
            'Tempo': audio_features['tempo'] if audio_features else None,
        }
        music_data.append(song_data)
    df = pd.DataFrame(music_data)
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
