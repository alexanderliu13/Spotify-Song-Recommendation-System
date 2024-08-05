import streamlit as st
from recommendation import get_playlist_data, get_access_token, hybrid_recommendations, get_playlist_name, get_playlist_image, get_song_image
import config
from sklearn.preprocessing import MinMaxScaler

CLIENT_ID = config.CLIENT_ID
CLIENT_SECRET = config.CLIENT_SECRET
token = get_access_token(CLIENT_ID, CLIENT_SECRET)

st.title("Spotify Music Recommendation System")
status = True
input_song_name = None
suggestion = True

playlist_link = st.text_input('Enter a Spotify Playlist URL')
# playlist_link = "https://open.spotify.com/playlist/4qpL3avso6uBuiUniFCIx2?si=028c95e47c3a4edd"
playlist_id = playlist_link[34: 56]

if playlist_id:
    playlist_name = get_playlist_name(playlist_id, token)
    playlist_image = get_playlist_image(playlist_id, token)
    music_df = get_playlist_data(playlist_id, token)
    st.write(f'Playlist Name: **{playlist_name}**')
    st.image(playlist_image, width = 300)
    # else:
    #     st.write("Failed to retrieve playlist data.")

    scaler = MinMaxScaler()
    music_features = music_df[['Danceability', 'Energy', 'Key', 'Loudness', 'Mode', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']].values
    music_features_scaled = scaler.fit_transform(music_features)

    input_song_name = st.text_input('Enter your song for suggestion:')
    if input_song_name:
        popularity_metric = st.slider(
            'Song Popularities',
            0, 100, 50)
        recommendations = hybrid_recommendations(input_song_name, music_df, music_features_scaled, num_recommendations=5, alpha=popularity_metric/100)
        if suggestion:
            st.write("Recommended Songs:")
            for index, row in recommendations.iterrows():
                url = row['External URLs']
                st.markdown(f"[**{row['Song Name']}**](%s)" % url + f"  \nRecommendation Score: {round(row['Recommendation Score'], 2)}")
                album_id = row["Album ID"]
                album_image = get_song_image(album_id, token)['images'][0]['url']
                st.image(album_image, width = 200)
                st.write("\n\n")
