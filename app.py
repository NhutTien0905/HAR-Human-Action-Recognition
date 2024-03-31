import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
import cv2
import numpy as np
import mediapipe as mp
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html
import dash
import base64
import threading
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytube import YouTube

label = "Warmup...."
n_time_steps = 10
warmup_frames = 60
lm_list = []

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

# model to cpu
# tf.config.set_visible_devices([], 'GPU')
model = tf.keras.models.load_model("model/model_0.h5")

cap = cv2.VideoCapture(0)

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        # print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return img


def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 0, 255)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img


def detect(model, lm_list):
    global label
    classes = ['CLAP', 'HIT', 'JUMP', 'RUN', 'SIT', 'STAND', 'THROW_HAND', 'WALK', 'WAVE_HAND']
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    results = model.predict(lm_list)
    label = str(classes[np.argmax(results)]) + " " + str(results[0][np.argmax(results)])
    return label

# Initialize Dash app
app = Dash(__name__)

# Initialize OpenCV capture
cap = cv2.VideoCapture(0)

# MediaPipe Pose initialization
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Define initial empty figure
fig_plotly = go.Figure()

# Function to convert OpenCV frame to base64 encoded image
def convert_frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

# Function to update the Plotly figure with skeleton keypoints
def update_figure(frame):
    global fig_plotly
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        # Extract skeleton keypoints
        keypoints = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark])

        # Clear existing traces
        fig_plotly = go.Figure()

        # Add each line segment of the skeleton as a scatter3d trace
        for connection in mp_pose.POSE_CONNECTIONS:
            start_point = keypoints[connection[0]]
            end_point = keypoints[connection[1]]
            fig_plotly.add_trace(go.Scatter3d(
                x=[start_point[0], end_point[0]],
                y=[start_point[1], end_point[1]],
                z=[-start_point[2], -end_point[2]],  # Negating z values to invert the axis (if needed)
                mode='lines',
                line=dict(color='blue', width=2)
            ))

        # Update layout
        fig_plotly.update_layout(scene=dict(
            xaxis=dict(range=[-1, 1], autorange=False),
            yaxis=dict(range=[-1, 1], autorange=False),
            zaxis=dict(range=[-1, 1], autorange=False),
        ))

# Function to stream a YouTube video
def stream_youtube_video(video_url):
    global cap, fig_plotly, lm_list, label
    yt = YouTube(video_url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    stream_url = stream.url

    capture = cv2.VideoCapture(stream_url)

    if not capture.isOpened():
        print("Error: Unable to open stream")
        return

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1st thread to update 3D skelecton plotly figure
        t2 = threading.Thread(target=update_figure, args=(frame,))
        t2.start()
        results = pose.process(img)
        print("Start detect....")

        if results.pose_landmarks:
            c_lm = make_landmark_timestep(results)

            lm_list.append(c_lm)
            if len(lm_list) == n_time_steps:
                # 2nd predict pose
                t1 = threading.Thread(target=detect, args=(model, lm_list,))
                t1.start()
                lm_list = []

            img = draw_landmark_on_image(mpDraw, results, img)

        img = draw_class_on_image(label, img)
        cv2.imshow('YouTube Stream', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Define app layout
app.layout = html.Div([
    html.Div([
        html.H3('Video Display'),
        
        html.Div([
            dcc.Input(id='input-box', type='text', value='0', style={'margin-right': '10px'}),
            html.Div(id="video-player"),
            html.Button('Get url', id='button-1', n_clicks=0, className="mr-2", style={'margin-right': '10px'}),
            html.Button('Cam', id='button-2', n_clicks=0, className="mr-2", style={'margin-right': '10px'}),
            # html.Button('Show skelection', id='button-3', n_clicks=0, className="mr-2", style={'margin-right': '10px'}),
        ], style={'text-align': 'center'}),
        html.Div(id='output-container-button'),
    ], style={'text-align': 'center'}),
    
    html.Div([
        html.Div(id='video-container', style={'margin-top': '20px', 'margin-bottom': '20px', 'display': 'inline-block', 'width': '50%'}),
        html.Div(dcc.Graph(id='skeleton-plot', figure=fig_plotly), style={'display': 'inline-block', 'width': '50%'})
    ]),
    
    dcc.Interval(id='interval-component', interval=300, n_intervals=0) 
], style={'text-align': 'center'})


# Callback to show YouTube video and skeleton plot
@app.callback(
    Output("video-player", "children"), # check xem dc ko
    [Input("button-1", "n_clicks")],
    [dash.dependencies.State('input-box', 'value')]
)
def update_video(n_clicks, video_url):
    if n_clicks > 0 and video_url:
        return html.Img(src=stream_youtube_video(video_url), style={'width': '100%'})
    elif not video_url:
        return html.Div("Enter a YouTube video URL to start streaming.")
    else:
        return None


# Callback to update the video and skeleton plot
@app.callback([Output('video-container', 'children'), Output('skeleton-plot', 'figure')],
              [Input('button-2', 'n_clicks'), Input('interval-component', 'n_intervals')])
def update_video_and_skeleton(button2_clicks, n_intervals):
    global cap, fig_plotly, lm_list, label
    if button2_clicks % 2 == 1:
        _, frame = cap.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1st thread to update 3D skelecton plotly figure
        t2 = threading.Thread(target=update_figure, args=(frame,))
        t2.start()
        results = pose.process(img)
        print("Start detect....")

        if results.pose_landmarks:
            c_lm = make_landmark_timestep(results)

            lm_list.append(c_lm)
            if len(lm_list) == n_time_steps:
                # 2nd predict pose
                t1 = threading.Thread(target=detect, args=(model, lm_list,))
                t1.start()
                lm_list = []

            img = draw_landmark_on_image(mpDraw, results, img)

        img = draw_class_on_image(label, img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_encoded = convert_frame_to_base64(img)
        video_element = html.Img(src='data:image/jpg;base64,{}'.format(frame_encoded))
        
        return video_element, fig_plotly
    return '', dash.no_update

# Run the Dash app
if __name__ == '__main__':
    app.run_server(port = 9900, debug=True)

