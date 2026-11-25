import requests
import streamlit as st
from spotipy.exceptions import SpotifyException
from recommendation import AUDIO_FEATURES, get_playlist_data, get_access_token, hybrid_recommendations, get_playlist_name, get_playlist_image, get_song_image, parse_playlist_id
import config
from sklearn.preprocessing import MinMaxScaler


CLIENT_ID = config.CLIENT_ID
CLIENT_SECRET = config.CLIENT_SECRET
token = get_access_token(CLIENT_ID, CLIENT_SECRET)

st.title("Spotify Song Recommender")

input_song_name = None

playlist_link = st.text_input('Enter a Spotify Playlist URL')
playlist_id = parse_playlist_id(playlist_link)

if playlist_link and not playlist_id:
    st.error('That does not look like a Spotify playlist URL. Paste a link like https://open.spotify.com/playlist/...')

if playlist_id:
    try:
        playlist_name = get_playlist_name(playlist_id, token)
        playlist_image = get_playlist_image(playlist_id, token)
        music_df = get_playlist_data(playlist_id, token)
    except SpotifyException as e:
        if e.http_status == 404:
            st.error('Spotify could not find that playlist. Playlists made by Spotify (e.g. Today\'s Top Hits) '
                     'are no longer available to new apps, so try a playlist made by a user.')
        else:
            st.error(f'Spotify returned an error ({e.http_status}) while loading the playlist.')
        st.stop()
    except requests.RequestException:
        st.error('Could not load audio features from ReccoBeats. Please try again later.')
        st.stop()
    st.write(f'Playlist Name: **{playlist_name}**')
    st.image(playlist_image, width = 300)

    skipped = music_df.attrs['total_tracks'] - len(music_df)
    if skipped > 0:
        st.caption(f'{skipped} track(s) were skipped because no audio features were available for them.')
    if len(music_df) < 2:
        st.error('Not enough songs in this playlist have audio features to make recommendations.')
        st.stop()

    scaler = MinMaxScaler()
    music_features = music_df[list(AUDIO_FEATURES.values())].values
    music_features_scaled = scaler.fit_transform(music_features)

    song_list = []
    for index, row in music_df.iterrows():
        song_list.append(row['Song Name'])
    input_song_name = st.selectbox('Select your song for suggestion:', song_list)

    if input_song_name:
        target_popularity = st.slider(
            'Target Popularity (0 = underground, 100 = mainstream)',
            0, 100, 50)
        recommendations = hybrid_recommendations(input_song_name, music_df, music_features_scaled, num_recommendations=5, target_popularity=target_popularity)
        st.write("Recommended Songs:")
        for index, row in recommendations.iterrows():
            url = row['External URLs']
            st.markdown(f"[**{row['Song Name']}**](%s)" % url + f"  \nRecommendation Score: {round(row['Recommendation Score'], 2)}")
            album_id = row["Album ID"]
            album_image = get_song_image(album_id, token)['images'][0]['url']
            st.image(album_image, width = 200)
            st.write("\n\n")
