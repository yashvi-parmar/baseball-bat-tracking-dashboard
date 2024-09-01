import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import ast
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from bat_averages import calculate_averages
# Tab
st.set_page_config(
    page_title="Bat Tracking Dashboard",
    page_icon="⚾️",
    layout="wide",
    initial_sidebar_state="expanded")

# Side bar
st.sidebar.header('Pages')
page = st.sidebar.selectbox('', ['Swing Analysis Tool', 'Comparison Tool', 'Info'])

# At-bat Events
if page == 'Swing Analysis Tool':
    st.title('Swing Analysis Tool')

    # Input Data
    df = pd.read_csv('90068.csv', index_col=0)

    # Filter Person and Event
    person_ids = df['personId.mlbId'].unique()
    selected_person_id = st.selectbox('Select a Person ID:', person_ids)
    df_person = df[df['personId.mlbId'] == selected_person_id]

    avg_swing = df_person['swing_length'].mean()
    avg_ev = df_person['hit_speed_mph'].mean()

    event_ids = df_person['event_id_updated'].unique()
    selected_event_id = st.selectbox('Select an Event ID:', event_ids)


    df= df_person[df_person['event_id_updated'] == selected_event_id]

    # Extract data from total_bat column
    def extract_positions(json_str):
            events = ast.literal_eval(json_str)
            head_positions = [event['head']['pos'] for event in events if 'head' in event]
            handle_positions = [event['handle']['pos'] for event in events if 'handle' in event]
            times = [event['time'] for event in events]
            return head_positions, handle_positions, times

    head_positions = []
    handle_positions = []
    times = []

    for json_str in df['total_bat']:
        head_pos, handle_pos, time = extract_positions(json_str)
        head_positions.extend(head_pos)
        handle_positions.extend(handle_pos)
        times.extend(time)

    positive_indices = [i for i, t in enumerate(times) if t >= 0]
    head_positions = [head_positions[i] for i in positive_indices]
    handle_positions = [handle_positions[i] for i in positive_indices]
    times = [times[i] for i in positive_indices]

    # Extract data from total_ball column
    def extract_positions(json_str):
            events = ast.literal_eval(json_str)
            positions = [event['pos'] for event in events if 'pos' in event]
            times = [event['time'] for event in events]
            return positions, times

    # Determine right handed or left handed
    def determine_handedness(handle_positions):
            x_coords = [pos[0] for pos in handle_positions]
            average_x = sum(x_coords) / len(x_coords)
            if average_x < 0:
                return 'Right-handed'
            else:
                return 'Left-handed'
    handedness = determine_handedness(handle_positions)

    # Head of the bat positions
    plot_df2 = pd.DataFrame(head_positions, columns=['X', 'Y', 'Z'])
    plot_df2['time'] = times

    # Handle of the bat position
    plot_df3 = pd.DataFrame(handle_positions, columns=['X', 'Y', 'Z'])
    plot_df3['time'] = times

    positions = []
    times = []

    for json_str in df['total_ball']:
        pos, time = extract_positions(json_str)
        positions.extend(pos)
        times.extend(time)

    # Ball Position 
    plot_df4 = pd.DataFrame(positions, columns=['X', 'Y', 'Z'])
    plot_df4['time'] = times

    # Just a bunch of variables
    time_hit = df['time_hit'].iloc[0]
    max_time = df['max_speed_time'].iloc[0]
    max_speed = df['max_speed_mph'].iloc[0]
    swing_length = df['swing_length'].iloc[0]
    exit_velocity = df['hit_speed_mph'].iloc[0]
    pitch_result = df['pitch_result'].iloc[0]
    runs_play = df['runs_play'].iloc[0]
    outs_play = df['outs_play'].iloc[0]
    angles = df['start.angle'].iloc[0]
    pitch_speed = df['pitch_speed_mph'].iloc[0]
    pitch_spin = df['pitch_spin_rpm'].iloc[0]
    hit_spin = df['hit_spin_rpm'].iloc[0]
    df['start.angle'] = df['start.angle'].apply(ast.literal_eval)
    df['spray_angle'] =df['start.angle'].apply(lambda x: x[0])
    df['launch_angle'] =df['start.angle'].apply(lambda x: x[1])
    df['ball_position'] = df['ball_position'].apply(ast.literal_eval)
    max_velo = df['max_velocity'].iloc[0]
    launch_angle = df['launch_angle'].iloc[0]

    # Get the ball trajectory before it was hit
    total_df = plot_df4[plot_df4['time'] < time_hit + 0.2]
    
    # Total Ball Trajectory
    trajectory = plot_df4
    # Ball trajectory before it was hit
    plot_df4 = plot_df4[plot_df4['time'] <= time_hit]
    # Head of bat at contact time
    plot_df22 = plot_df2[plot_df2['time'] == time_hit]

    # Create a new dataframe to store information about the position of the bat at contact with the ball
    bat_df = pd.merge(plot_df22, plot_df3, on='time', how='inner')

    # New dataframe to show the result of the game 
    new_df = pd.DataFrame()
    new_df = df[['pitch_result', 'runs_game_team1',
        'runs_game_team2',
        'runs_play', 'outs_play', 'balls_plateAppearance',
        'balls_play', 'strikes_plateAppearance', 'strikes_play']]
    new_df.columns = [
        'Pitch Result', 'Game Runs Team 1', 'Game Runs Team 2',
        'Runs Play',
        'Outs Play', 'Balls Plate Appearance',
        'Balls Play', 'Strikes Plate Appearance', 'Strikes Play'
    ]

    plot_df4['dX'] = plot_df4['X'].diff()
    plot_df4['dY'] = plot_df4['Y'].diff()
    plot_df4['dZ'] = plot_df4['Z'].diff()
    last_dx = plot_df4['dX'].iloc[-1]
    last_dy = plot_df4['dY'].iloc[-1]
    last_dz = plot_df4['dZ'].iloc[-1]

    last_horizontal_displacement = np.sqrt(last_dx**2 + last_dy**2)

    # Finding the angle of the incoming pitch 
    last_approach_angle = np.degrees(np.arctan(last_dz / last_horizontal_displacement))
    
    # First Row
    col = st.columns((2, 2), gap='small')
    average_bat_speed, average_VAA_deg, average_HAA_deg, average_pitch = calculate_averages(df_person)


    # Give Player Info
    with col[0]:
        with st.expander("Player Info"):
            data = {
                "Team Id": [df['teamId.mlbId'].iloc[0]],
                "Handedness": [handedness],
                "Average Swing Length": [f"{avg_swing:.2f} ft"],
                "Average Exit Velocity": [f"{avg_ev:.2f} degrees"],
                "Average Bat Speed at Contact": [f"{average_bat_speed:.2f} mph"],
                "Average Vertical Attack Angle (VAA)": [f"{average_VAA_deg:.2f} degrees"],
                "Average Pitch Approach Angle": [f"{average_pitch:.2f} degrees"]
            }

            df_info = pd.DataFrame(data)
            st.table(df_info)
    # Display game score
    with col[1]:
        with st.expander("Game Score"):  
            st.table(new_df)

    # Rounded variables
    max_speed = round(max_speed, 2)
    max_time_rounded= round(max_time, 2)
    time_hit_rounded = round(time_hit, 2)
    exit_velocity= round(exit_velocity, 2)
    swing_length = round(swing_length, 2)
    pitch_speed = round(pitch_speed, 2)
    pitch_spin = round(pitch_spin, 2)
    hit_spin = round(hit_spin, 2)

    ball_df = df['ball_points']
    df['head_hit'] = df['head_hit'].apply(lambda x: json.loads(x.replace("'", "\"")))
    df['handle_hit'] = df['handle_hit'].apply(lambda x: json.loads(x.replace("'", "\"")))
    t_hit = time_hit 
  

    # Finding the position of the ball on the bat at contact
    def parse_ball_data(data_str):
        return ast.literal_eval(data_str)
    parsed_data = ball_df.apply(parse_ball_data)
    flattened_data = []
    for entry in parsed_data:
        for item in entry:
            flattened_data.append(item)
    flattened_df = pd.DataFrame(flattened_data)
    flattened_df = flattened_df.sort_values('time').reset_index(drop=True)
    t1, t2 = None, None
    pos1, pos2 = None, None

    for i in range(len(flattened_df) - 1):
        if flattened_df.loc[i, 'time'] <= t_hit <= flattened_df.loc[i + 1, 'time']:
            t1 = flattened_df.loc[i, 'time']
            pos1 = flattened_df.loc[i, 'pos']
            t2 = flattened_df.loc[i + 1, 'time']
            pos2 = flattened_df.loc[i + 1, 'pos']
            break

    if t1 is not None and t2 is not None and pos1 is not None and pos2 is not None:
        x1, y1, z1 = pos1
        x2, y2, z2 = pos2

        x_hit = x1 + (t_hit - t1) / (t2 - t1) * (x2 - x1)
        y_hit = y1 + (t_hit - t1) / (t2 - t1) * (y2 - y1)
        z_hit = z1 + (t_hit - t1) / (t2 - t1) * (z2 - z1)

        ball_position_at_hit = (x_hit, y_hit, z_hit)
    
        bat_head_pos = df['head_hit'].iloc[0]['pos']
        bat_handle_pos = df['handle_hit'].iloc[0]['pos']

        bat_head_pos = np.array(bat_head_pos)
        bat_handle_pos = np.array(bat_handle_pos)
        ball_position_at_hit = np.array(ball_position_at_hit)

        bat_vector = bat_head_pos - bat_handle_pos
        bat_unit_vector = bat_vector / np.linalg.norm(bat_vector)

        t = np.dot(ball_position_at_hit - bat_handle_pos, bat_unit_vector) / np.dot(bat_vector, bat_unit_vector)
        intersection_point = bat_handle_pos + t * bat_vector


    # Merge data to get one dataframe with all the bat's positional coordinates
    bat = plot_df2.merge(plot_df3[['time', 'X', 'Y', 'Z']], on = 'time', how='left')
    bat.rename(columns={'X_x': 'x_head', 'Y_x': 'y_head', 'Z_x': 'z_head', 'X_y': 'x_handle','Y_y': 'y_handle', 'Z_y': 'z_handle'}, inplace=True)
    closest_idx = (total_df['time'] - time_hit).abs().idxmin()

    # Select the three points before the time_hit
    new_df = total_df.iloc[max(0, closest_idx-6):closest_idx]
    bat = bat[(bat['time'] > 0.15) & (bat['time'] < 0.7)]
    bat = bat.reset_index(drop=True)

    bat_df = bat
    ball_df = total_df

    closest_idx = (total_df['time'] - time_hit).abs().idxmin()

    # Transfer back to df
    df = bat

    # Calculate the differences
    df['dx'] = df['x_head'] - df['x_handle']
    df['dy'] = df['y_head'] - df['y_handle']
    df['dz'] = df['z_head'] - df['z_handle']

    # Find the length of the bat
    df['bat_length'] = np.sqrt(df['dx']**2 + df['dy']**2 + df['dz']**2)

    df['dx_norm'] = df['dx'] / df['bat_length']
    df['dy_norm'] = df['dy'] / df['bat_length']
    df['dz_norm'] = df['dz'] / df['bat_length']

    df['x_sweet_spot'] = df['x_head'] - 0.5 * df['dx_norm']
    df['y_sweet_spot'] = df['y_head'] - 0.5 * df['dy_norm']
    df['z_sweet_spot'] = df['z_head'] - 0.5 * df['dz_norm']

    df['vx'] = df['x_sweet_spot'].diff() / df['time'].diff()
    df['vy'] = df['y_sweet_spot'].diff() / df['time'].diff()
    df['vz'] = df['z_sweet_spot'].diff() / df['time'].diff()

    df = df.dropna().reset_index(drop=True)
    df['v_resultant'] = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)

    # Calculating the Vertical Attack Angle 
    df['VAA_rad'] = np.arctan2(df['vz'], np.sqrt(df['vx']**2 + df['vy']**2))
    df['VAA_deg'] = np.degrees(df['VAA_rad'])

    attack_angle = df[df['time'] == time_hit]
    vaa = attack_angle['VAA_deg'].iloc[0]

    # Calculate the Horizontal Attack Angle
    df['perp_x'] = -df['dz_norm']
    df['perp_z'] = df['dx_norm']

    df['HAA_rad'] = np.arctan2(df['perp_x'], df['perp_z'])
    df['HAA_deg'] = np.degrees(df['HAA_rad'])
    haaat = df[df['time'] == time_hit]
    haa = haaat['HAA_deg'].iloc[0]

    intersection_time = time_hit


    # Find the bat head and handle positions at the intersection time using interpolation
    x_head_intersection = np.interp(intersection_time, df['time'], df['x_head'])
    y_head_intersection = np.interp(intersection_time, df['time'], df['y_head'])
    z_head_intersection = np.interp(intersection_time, df['time'], df['z_head'])

    x_handle_intersection = np.interp(intersection_time, df['time'], df['x_handle'])
    y_handle_intersection = np.interp(intersection_time, df['time'], df['y_handle'])
    z_handle_intersection = np.interp(intersection_time, df['time'], df['z_handle'])
    head_position_row = df[df['time'] == time_hit]

    # Extract the head position coordinates
    head_position = head_position_row[['x_head', 'y_head', 'z_head']].values[0]

    # Coordinates of the intersection point
    intersection_x, intersection_y, intersection_z = intersection_point[0], intersection_point[1], intersection_point[2]

    def interpolate_positions(time, df, columns):
        return {col: np.interp(time, df['time'], df[col]) for col in columns}

    # Calculate Euclidean distance
    def euclidean_distance(point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    # Calculate bat angle
    def calculate_bat_angle(df):
        df['dx'] = df['x_head'] - df['x_handle']
        df['dy'] = df['y_head'] - df['y_handle']
        df['dz'] = df['z_head'] - df['z_handle']
        df['horizontal_distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
        df['bat_angle'] = np.arctan2(df['dz'], df['horizontal_distance']) * (180 / np.pi)

    # Calculate bat speed
    def calculate_bat_speed(df):
        df['head_distance'] = np.sqrt(df[['x_head', 'y_head', 'z_head']].diff().pow(2).sum(axis=1))
        df['time_diff'] = df['time'].diff()
        df['head_speed'] = df['head_distance'] / df['time_diff'] * 0.681818
        return df.dropna()

    # Main processing
    columns = ['x_head', 'y_head', 'z_head', 'x_handle', 'y_handle', 'z_handle']
    intersection_pos = interpolate_positions(intersection_time, df, columns)
    time_hit_pos = interpolate_positions(time_hit, df, columns)

    head_position = df.loc[df['time'] == time_hit, ['x_head', 'y_head', 'z_head']].values[0]
    distance_inches = euclidean_distance(np.array(intersection_point), head_position) * 12

    calculate_bat_angle(df)
    attack_angle = df.loc[df['time'] == time_hit, 'VAA_deg'].iloc[0]

    bat_vector = np.array([time_hit_pos['x_head'] - time_hit_pos['x_handle'], time_hit_pos['y_head'] - time_hit_pos['y_handle']])
    bat_angle = np.degrees(np.arctan2(bat_vector[1], bat_vector[0]))

    df = calculate_bat_speed(df)

    lowest_point_idx = df['z_head'].idxmin()
    lowest_point = df.loc[lowest_point_idx]
    

    def get_launch_result(angle):
        if angle < 8:
            return 'Burner', 'red'
        elif 8 <= angle < 32:
            return 'Sweet Spot', 'green'
        elif 32 <= angle < 40:
            return 'Flare', 'yellow'
        else:
            return 'Hit Under', 'red'

    def get_hit_type(angle):
        if angle < 10:
            return 'Ground Ball', 'red'
        elif 10 <= angle < 25:
            return 'Line Drive', 'green'
        elif 25 <= angle < 50:
            return 'Fly Ball', 'yellow'
        else:
            return 'Pop Fly', 'red'

    def get_distance_color(distance):
        if 4 < distance < 7:
            return 'green'
        elif 7 < distance < 9:
            return 'yellow'
        else:
            return 'red'

    def get_velo_color(diff):
        if abs(diff) <= 5:
            return 'green'
        elif abs(diff) <= 10:
            return 'yellow'
        else:
            return 'red'

    def get_swing_type(vaa):
        if vaa >= 5:
            return 'Upper Cut', 'green'
        elif vaa < 0:
            return 'Under Cut', 'red'
        else:
            return 'Level', 'yellow'

    def get_exit_velocity_color(velocity):
        if velocity >= 80:
            return 'green'
        elif 80 <= velocity < 80:
            return 'yellow'
        else:
            return 'red'

    result, color_launch = get_launch_result(launch_angle)
    hit_type, result_color = get_hit_type(launch_angle)
    distance_color = get_distance_color(distance_inches)

    on_plane = vaa + last_approach_angle
    color_velo = get_velo_color(on_plane)

    swing, color = get_swing_type(vaa)
    color_exit_velocity = get_exit_velocity_color(exit_velocity)

    # Styling for the boxes
    st.markdown("""
    <style>
    .custom-box {
        border: 2px solid white;
        width: 150px; /* Set the width */
        height: 150px; /* Set the height to match the width to make it square */

        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        margin-bottom: 20px; /* Add space at the bottom */
        flex-direction: column; /* Stack elements vertically */
    }

    .box-launch {
        font-size: 3em; /* Larger font for the number */
        margin-bottom: auto; /* Pushes the number to the top */
        color: {color_launch};
    }
    .box-number {
        font-size: 3em; /* Larger font for the number */
        margin-bottom: auto; /* Pushes the number to the top */
        
    }
    .box-value {
        font-size: 2em; /* Larger font for the number */
        margin-bottom: auto; /* Pushes the number to the top */
    }

    .box-label {
        font-size: 1em; /* Smaller font for the label */
        margin-top: auto; /* Pushes the label to the bottom */
    }
    .centered-columns {
        display: flex;
        justify-content: center;
        gap: 10px; /* Adjust spacing between boxes */
        align-items: center;
    }
    .streamlit-expanderHeader {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5, col6 = st.columns([1,1,1,1,1, 1], gap="small")

    with col1:
        st.markdown(f'''
        <div class="custom-box">
            <div class="box-launch" style="color:{color_launch}" >{launch_angle:.2f}</div>
            <div class="box-label">Launch Angle (degrees)</div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="custom-box" ><div class="box-number" style="color:{color_velo}" >{on_plane:.2f}</div><div class="box-label">On Plane (degrees)</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="custom-box"><div class="box-number" style="color:{distance_color}">{distance_inches:.2f}</div><div class="box-label">Contact From Head of the Bat (inches)</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="custom-box"><div class="box-number" style="color: {color_exit_velocity}">{exit_velocity:.2f}</div><div class="box-label">Exit Velocity (mph)</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="custom-box"><div class="box-value" style="color:{color}">{swing}</div><div class="box-label">Swing</div></div>', unsafe_allow_html=True)
    with col6:
        st.markdown(f'<div class="custom-box"><div class="box-value" style="color:{result_color}">{result}</div><div class="box-label">Contact</div></div>', unsafe_allow_html=True)


    col = st.columns((2,3), gap='medium')

    # Line graphs
    with col[0]:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df['time'],
            y=df['head_speed'],
            mode='lines+markers',
            name='Bat Speed',
            line=dict(color='blue'),
            marker=dict(size=6)
        ))

        fig1.add_trace(go.Scatter(
            x=[time_hit],
            y=[df.loc[df['time'] == time_hit, 'head_speed'].values[0]],
            mode='markers',
            name='Time Hit',
            marker=dict(color='red', size=6)
        ))

        fig1.update_layout(
            title='Bat Speed vs Time',
            xaxis_title='Time',
            yaxis_title='Bat Speed (mph)',
            template='plotly_white',
            height=300  
        )

        st.plotly_chart(fig1, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df['time'],
            y=df['VAA_deg'],
            mode='lines+markers',
            name='Attack Angle',
            line=dict(color='blue'),
            marker=dict(size=6)
        ))
        fig2.add_trace(go.Scatter(
            x=[time_hit],
            y=[df.loc[df['time'] == time_hit, 'VAA_deg'].values[0]],
            mode='markers',
            name='Time Hit',
            marker=dict(color='red', size=6)
        ))

        fig2.update_layout(
            title='Vertical Attack Angle vs Time',
            xaxis_title='Time',
            yaxis_title='Attack Angle',
            template='plotly_white',
            height=300 
        )

        st.plotly_chart(fig2, use_container_width=True)

    # Model the swing path
    with col[1]:
        fig = go.Figure()

        
        fig.add_trace(go.Scatter3d(
            x=df['x_head'], 
            y=df['y_head'], 
            z=df['z_head'], 
            mode='lines',
            marker=dict(size=2, color='purple'),
            name='Head Positions',
            hoverinfo='text',
            text=df['time']
        ))

        fig.add_trace(go.Scatter3d(
            x=df['x_handle'], 
            y=df['y_handle'], 
            z=df['z_handle'], 
            mode='lines',
            marker=dict(size=2, color='purple'),
            name='Handle Positions',
            hoverinfo='text',
            text=df['time']
        ))
        fig.add_trace(go.Scatter3d(
            x=df['x_head'], 
            y=df['y_head'], 
            z=df['z_head'], 
            mode='lines',
            marker=dict(size=2, color='purple'),
            name='Head Positions',
            hoverinfo='text',
            text=df['time']
        ))
        fig.add_trace(go.Scatter3d(
            x=[intersection_point[0]],
            y=[intersection_point[1]],
            z=[intersection_point[2]],
            mode='markers',
            marker=dict(size=5, color='gray'),
            name='Ball Made Contact',
            hoverinfo='text',
            text=f'Contact Distance from Head: {distance_inches:.2f} inches'
        ))
        fig.add_annotation(
            x=1,
            y=0.6,
            text=f'Contact Distance from Head: {distance_inches:.2f} inches',
            showarrow=False,
            xref='paper',
            yref='paper',
            xanchor='left',
            yanchor='bottom',
            font=dict(size=12)
    )


        frames = []

        for t in sorted(set(df['time'])):
            lines_x, lines_y, lines_z = [], [], []

            for i in range(len(df)):
                if df['time'].iloc[i] <= t:
                    line_color = 'orange' if df['time'].iloc[i] == time_hit else 'green'
                    lines_x.extend([df.iloc[i]['x_head'], df.iloc[i]['x_handle'], None])
                    lines_y.extend([df.iloc[i]['y_head'], df.iloc[i]['y_handle'], None])
                    lines_z.extend([df.iloc[i]['z_head'], df.iloc[i]['z_handle'], None])
                
            frame_data = [
                go.Scatter3d(
                    x=lines_x,
                    y=lines_y,
                    z=lines_z,
                    mode='lines',
                    line=dict(width=3, color='purple'),
                    showlegend=False
                )
            ]

            current_head_speed = df.loc[df['time'] <= t, 'head_speed'].values[-1]
            current_attack_angle = df.loc[df['time'] <= t, 'VAA_deg'].values[-1]
            current_bat_angle = df.loc[df['time'] <= t, 'bat_angle'].values[-1]
    
            frames.append(go.Frame(data=frame_data, name=str(t), 
                                layout=go.Layout(annotations=[
                                        go.layout.Annotation(
                                            text=f"Swing Length: {swing_length:.2f} ft<br>Exit Velocity: {exit_velocity:.2f} mph<br>Attack Angle at Contact: {vaa:.2f}°<br>Pitch Approach Angle: {last_approach_angle:.2f}°<br>Result: {hit_type}",
                                            x=1.0,
                                            y=0.5,
                                            xref="paper",
                                            yref="paper",
                                            showarrow=False,
                                            font=dict(size=12),
                                            borderpad=4  
                                        )
                                    , 
                                    go.layout.Annotation(
                                        text=f"Bat Speed: {current_head_speed:.2f} mph<br>Vertical Attack Angle: {current_attack_angle:.2f}°<br>Vertical Bat Angle: {current_bat_angle:.2f}°",
                                        x=1.28,  
                                        y=0.3,
                                        xref="paper",
                                        yref="paper",
                                        showarrow=False,
                                        font=dict(size=12),
                                        borderpad=4  
                                    )
                                ]))
            
            )

        fig.update(frames=frames)

        time_hit_index = df.index[df['time'] == time_hit].tolist()[0]
        fig.add_trace(go.Scatter3d(
            x=[df['x_head'][time_hit_index], df['x_handle'][time_hit_index]],
            y=[df['y_head'][time_hit_index], df['y_handle'][time_hit_index]],
            z=[df['z_head'][time_hit_index], df['z_handle'][time_hit_index]],
            mode='lines',
            line=dict(color='white', width=3),
            name='Time Ball Made Contact with Bat'
        ))

        fig.update_layout(
            annotations=[
            go.layout.Annotation(
                text=f"Swing Length: {swing_length:.2f} ft<br>Exit Velocity: {exit_velocity:.2f} mph<br>Attack Angle at Contact: {vaa:.2f}°<br>Pitch Approach Angle: {last_approach_angle:.2f}°<br>Result: {hit_type}",
                x=1.0,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12),
                
            )
        ], 
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, title=''),
                yaxis=dict(showgrid=False, showticklabels=False, title=''),
                zaxis=dict(showgrid=False, showticklabels=False, title=''),
                camera=dict(
                    eye=dict(x=1, y=1, z=0)  
                )
            ),
            legend_title='Legend',
            
            height=600,
            width=600,
            margin=dict(l=0, r=0, b=10, t=10),
            updatemenus=[{
                'buttons': [
                    {
                        'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
                        'label': 'Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                        'label': 'Pause',
                        'method': 'animate'
                    }
                ],
                'direction': 'right',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'type': 'buttons',
                'x': 0.25,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top', 
            
            }],
            sliders=[{
                'steps': [{'args': [[str(time)], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate'}],
                        'label': str(time), 'method': 'animate'} for time in sorted(set(df['time']))],
                'transition': {'duration': 50},
                'x': 0.3,
                'len': 0.6,
                'currentvalue': {'font': {'size': 20}, 'prefix': 'Time: ', 'visible': True, 'xanchor': 'center'},
                'pad': {'b': 10, 't': 30},
                'xanchor': 'left',
                'yanchor': 'top'
            }]
        )

        fig.add_trace(go.Scatter3d(
            x=[lowest_point['x_head']],
            y=[lowest_point['y_head']],
            z=[lowest_point['z_head']],
            mode='markers+text',
            marker=dict(size=5, color='blue'),
            name= "Lowest Point",
            showlegend=True
        ))
        
        st.plotly_chart(fig, use_container_width=True)

    # Horizontal Attack Angle Graph
    col = st.columns((2, 2), gap='small')
    with col[0]:
        with st.expander("Horizontal Attack Angle vs Time"):
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df['time'],
                y=df['HAA_deg'],
                mode='lines+markers',
                name='Attack Angle',
                line=dict(color='blue'),
                marker=dict(size=6)
            ))

            fig2.add_trace(go.Scatter(
                x=[time_hit],
                y=[df.loc[df['time'] == time_hit, 'HAA_deg'].values[0]],
                mode='markers',
                name='Time Hit',
                marker=dict(color='red', size=6)
            ))

            fig2.update_layout(
                title='Horiztonal Attack Angle vs Time',
                xaxis_title='Time',
                yaxis_title='Attack Angle',
                template='plotly_white',
                height=300 
            )

            st.plotly_chart(fig2, use_container_width=True)

    # Horizontal Attack Angle Animation 
    with col[1]:
        with st.expander("Horizontal Attack Angle"):

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['x_head'],
                y=df['y_head'],
                mode='lines',
                name='Bat Head',
                line=dict(color='blue', width=4) 
            ))
            fig.add_trace(go.Scatter(
                x=df['x_handle'],
                y=df['y_handle'],
                mode='lines',
                name='Bat Handle',
                line=dict(color='red', width=4) 
            ))

            fig.add_trace(go.Scatter(
                x=[x_handle_intersection, x_head_intersection],
                y=[y_handle_intersection, y_head_intersection],
                mode='lines+markers',
                name='Bat Position at time_hit',
                line=dict(color='white', width=4),
                marker=dict(size=8, color='white')
            ))
           

            angle_length = 0.2
            angle_x = [x_handle_intersection, x_handle_intersection + angle_length * np.cos(np.radians(bat_angle))]
            angle_y = [y_handle_intersection, y_handle_intersection + angle_length * np.sin(np.radians(bat_angle))]

         

            x_min = min(df['x_head'].min(), df['x_handle'].min())
            x_max = max(df['x_head'].max(), df['x_handle'].max())
            y_min = min(df['y_head'].min(), df['y_handle'].min())
            y_max = max(df['y_head'].max(), df['y_handle'].max())
            frames = []
            for t in sorted(set(df['time'])):
                lines_x, lines_y = [], []

                for i in range(len(df)):
                    if df['time'].iloc[i] <= t:
                        line_color = 'orange' if df['time'].iloc[i] == time_hit else 'purple'
                        lines_x.extend([df.iloc[i]['x_head'], df.iloc[i]['x_handle'], None])
                        lines_y.extend([df.iloc[i]['y_head'], df.iloc[i]['y_handle'], None])

                current_haa = df.loc[df['time'] <= t, 'HAA_deg'].values[-1]

                frames.append(go.Frame(
                    data=[go.Scatter(
                        x=lines_x,
                        y=lines_y,
                        mode='lines',
                        line=dict(color=line_color, width=2, dash='dot'),
                        showlegend=False
                    )],
                    name=f'frame_{t}',
                    layout=go.Layout(
                        annotations=[
                            
                            dict(
                                x=x_min + (x_max - x_min) * 0.05, 
                                y=y_min + (y_max - y_min) * 0.05,
                                text=f"Horizontal Attack Angle = {current_haa:.2f}°",
                                showarrow=False,
                                font=dict(size=12, color='red')
                            ),
                            
                        ]
                    )
                ))
            fig.update_layout(
            
                sliders=[{
                    'steps': [{'method': 'animate', 'args': [[f'frame_{t}'], {'mode': 'immediate', 'frame': {'duration': 500, 'redraw': True}, 'transition': {'duration': 0}}], 'label': str(t)} for t in sorted(set(df['time']))],
                    'transition': {'duration': 0},
                    'x': 0.1,
                    'xanchor': 'left',
                    'y': -0.3,
                    'yanchor': 'top'
                }]
            )

            fig.update(frames=frames)
            fig.update_layout(
                xaxis_title='X',
                yaxis_title='Y',
                legend=dict(
                    x=0,
                    y=1
                ),
                width=800,
                height=600,  
                xaxis=dict(range=[x_min, x_max]),
                yaxis=dict(range=[y_min, y_max])
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Pitch Plots 
    col = st.columns((2, 2), gap='medium')
    # The trajectory of the ball before it was hit
    with col[0]:
        with st.expander("Pitch Trajectory"):
            st.write("## Pitch Info")
            st.write(f"Pitch Speed: {pitch_speed} mph")
            st.write(f"Pitch Spin: {pitch_spin} rpm")
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                    x=plot_df4['X'],
                    z=plot_df4['Z'],  
                    y=-plot_df4['Y'],  
                    mode='markers+lines',
                    marker=dict(size=4, color=plot_df4['time'], showscale=True),
                    
                    line=dict(color='blue', width=2),
                    name='Trajectory',
                    hoverinfo='text',
                    text=plot_df4['time']
                ))

            width_in_feet = 17 / 12 
            height_in_feet = 3.5 
            bottom_of_strike_zone = 1.5 
            #Strike zone - assumed avg height of batter
            strike_zone_x = [width_in_feet / 2, width_in_feet / 2, -width_in_feet / 2, -width_in_feet / 2, width_in_feet / 2]
            strike_zone_z = [bottom_of_strike_zone, bottom_of_strike_zone + height_in_feet, bottom_of_strike_zone + height_in_feet, bottom_of_strike_zone, bottom_of_strike_zone]
            strike_zone_y = [1, 1, 1, 1, 1]

            fig.add_trace(go.Scatter3d(
                    x=strike_zone_x,
                    z=strike_zone_z,
                    y=[0, 0, 0, 0, 0],  
                    mode='lines',
                    line=dict(color='red', width=4),
                    name='Strike Zone'
                ))
            fig.update_layout(
                    scene=dict(
                        xaxis=dict(title='X Coordinate (Width)', range=[-3, 3]),
                        zaxis=dict(title='Y Coordinate (Height)', range=[0, 10]),
                        yaxis=dict(title='Z Coordinate (Depth)', range=[-50, 2]),
                        camera=dict(eye=dict(x=-2, y=2, z=0)) 
                    ),
                    legend=dict(
                    title=dict(text='Event'),
                    x=0, 
                    y=1,
                    traceorder='normal',
                    bgcolor='rgba(0,0,0,0)'
                ),margin=dict(l=0, r=0, b=0, t=0),
                    height=500
                )

            st.plotly_chart(fig, use_container_width=True)
        # The whole ball trajectory
        with col[1]:
            with st.expander("Ball Trajectory"):
                fig = go.Figure()
                fig.add_trace(go.Scatter3d(
                        x=trajectory['X'],
                        z=trajectory['Z'],  
                        y=-trajectory['Y'], 
                        mode='markers+lines',
                        marker=dict(size=4, color=trajectory['time'], showscale=True),
                        
                        line=dict(color='blue', width=2),
                        name='Trajectory',
                        hoverinfo='text',
                        text=trajectory['time']
                    ))
                fig.update_layout(
                    scene=dict(
                        camera=dict(eye=dict(x=-2, y=2, z=0)) 
                    ),
                    
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)

# Compare Players Page
elif page == 'Comparison Tool':
    st.title('Comparison Tool')
    # Ingest data
    df = pd.read_csv('90068.csv', index_col=0)

    col1, col2 = st.columns(2)

    with col1:
        person_ids = df['personId.mlbId'].unique()
        selected_person_id_A = st.selectbox('Select Player 1:', person_ids)
        df_A = df[df['personId.mlbId'] == selected_person_id_A]
        event_ids_A = df_A['event_id_updated'].unique()
        selected_event_id_A = st.selectbox('Select an Event ID for Player 1:', event_ids_A)

    with col2:
        # Allow selection of the same player or a different player
        selected_person_id_B = st.selectbox('Select Player 2:', person_ids)
        df_B = df[df['personId.mlbId'] == selected_person_id_B]
        event_ids_B = df_B['event_id_updated'].unique()

        # If the same player is selected, filter out the selected_event_id_A from event_ids_B
        if selected_person_id_A == selected_person_id_B:
            event_ids_B = [eid for eid in event_ids_B if eid != selected_event_id_A]

        selected_event_id_B = st.selectbox('Select an Event ID for Player 2:', event_ids_B)

    def extract_positions(json_str, keys=['head', 'handle']):
        events = ast.literal_eval(json_str)
        head_positions = [event[keys[0]]['pos'] for event in events if keys[0] in event]
        handle_positions = [event[keys[1]]['pos'] for event in events if keys[1] in event]
        times = [event['time'] for event in events]
        return head_positions, handle_positions, times

    def extract_ball_positions(json_str):
        events = ast.literal_eval(json_str)
        positions = [event['pos'] for event in events if 'pos' in event]
        times = [event['time'] for event in events]
        return positions, times

    def filter_positive_times(positions, times):
        positive_indices = [i for i, t in enumerate(times) if t >= 0]
        filtered_positions = [positions[i] for i in positive_indices]
        filtered_times = [times[i] for i in positive_indices]
        return filtered_positions, filtered_times

    def determine_handedness(handle_positions):
        x_coords = [pos[0] for pos in handle_positions]
        average_x = sum(x_coords) / len(x_coords)
        return 'Right-handed' if average_x < 0 else 'Left-handed'

    def process_positions(df, column, keys=['head', 'handle']):
        head_positions, handle_positions, times = [], [], []
        for json_str in df[column]:
            head_pos, handle_pos, time = extract_positions(json_str, keys)
            head_positions.extend(head_pos)
            handle_positions.extend(handle_pos)
            times.extend(time)
        head_positions, times = filter_positive_times(head_positions, times)
        handle_positions, _ = filter_positive_times(handle_positions, times)
        return head_positions, handle_positions, times

    # Main function
    def main(df_person1, df_person2):
        
        head_positions1, handle_positions1, times1 = process_positions(df_person1, 'total_bat')
        handedness1 = determine_handedness(handle_positions1)

        head_positions2, handle_positions2, times2 = process_positions(df_person2, 'total_bat')
        handedness2 = determine_handedness(handle_positions2)

        def extract_positions(json_str):
                events = ast.literal_eval(json_str)
                head_positions = [event['head']['pos'] for event in events if 'head' in event]
                handle_positions = [event['handle']['pos'] for event in events if 'handle' in event]
                times = [event['time'] for event in events]
                return head_positions, handle_positions, times

        head_positions = []
        handle_positions = []
        times = []

        for json_str in df_person1['total_bat']:
            head_pos, handle_pos, time = extract_positions(json_str)
            head_positions.extend(head_pos)
            handle_positions.extend(handle_pos)
            times.extend(time)

        positive_indices = [i for i, t in enumerate(times) if t >= 0]
        head_positions = [head_positions[i] for i in positive_indices]
        handle_positions = [handle_positions[i] for i in positive_indices]
        times = [times[i] for i in positive_indices]


        def extract_positions(json_str):
                events2 = ast.literal_eval(json_str)
                head_positions2 = [event2['head']['pos'] for event2 in events2 if 'head' in event2]
                handle_positions2 = [event2['handle']['pos'] for event2 in events2 if 'handle' in event2]
                times2 = [event2['time'] for event2 in events2]
                return head_positions2, handle_positions2, times2

        head_positions2 = []
        handle_positions2 = []
        times2 = []

        for json_str in df_person2['total_bat']:
            head_pos, handle_pos, time = extract_positions(json_str)
            head_positions2.extend(head_pos)
            handle_positions2.extend(handle_pos)
            times2.extend(time)

        positive_indices2 = [i for i, t in enumerate(times2) if t >= 0]
        head_positions2 = [head_positions2[i] for i in positive_indices2]
        handle_positions2 = [handle_positions2[i] for i in positive_indices2]
        times2 = [times2[i] for i in positive_indices2]

        time_hit_1 = df_person1['time_hit'].iloc[0]
        swing_length_1 = df_person1['swing_length'].iloc[0]
        exit_velocity_1 = df_person1['hit_speed_mph'].iloc[0]
        hit_spin_1 = df_person1['hit_spin_rpm'].iloc[0]
        df_person1['start.angle'] = df_person1['start.angle'].apply(ast.literal_eval)
        df_person1['spray_angle'] =df_person1['start.angle'].apply(lambda x: x[0])
        df_person1['launch_angle'] =df_person1['start.angle'].apply(lambda x: x[1])
        launch_angle_1 = df_person1['launch_angle'].iloc[0]


        time_hit_2 = df_person2['time_hit'].iloc[0]
        swing_length_2 = df_person2['swing_length'].iloc[0]
        exit_velocity_2 = df_person2['hit_speed_mph'].iloc[0]
        hit_spin2 = df_person2['hit_spin_rpm'].iloc[0]
        df_person2['start.angle'] = df_person2['start.angle'].apply(ast.literal_eval)
        df_person2['spray_angle'] =df_person2['start.angle'].apply(lambda x: x[0])
        df_person2['launch_angle'] =df_person2['start.angle'].apply(lambda x: x[1])
        launch_angle_2 = df_person2['launch_angle'].iloc[0]

        col = st.columns((2, 2), gap='medium')
        # Display Player 1 and Player 2 Info
        with col[0]:
            st.text(f"Team 1 Id: {df_person1['teamId.mlbId'].iloc[0]}")
            st.text(f"Handedness: {handedness1}")
            with st.expander("Swing Info"):
                    st.write(f"Time of Hit: {time_hit_1:.2f}s")
                    st.write(f"Exit Velocity: {exit_velocity_1:.2f} mph")
                    st.write(f"Hit Spin: {hit_spin_1:.2f} rpm")
                    st.write(f"Swing Length: {swing_length_1:.2f} ft")
                    st.write(f"Launch Angle: {launch_angle_1:.2f} degrees")
                    if launch_angle_1 < 10:
                        result = 'ground ball'
                        a = 1
                    elif 10 <= launch_angle_1 < 25:
                            result = 'line drive'
                            a = 3
                    elif 25 <= launch_angle_1 < 50:
                            result = 'fly ball'
                            a = 2
                    else:
                        result = 'pop up'
                    st.write(f"The ball was a {result}")
                

        
        with col[1]:
            st.text(f"Team 2 Id: {df_person2['teamId.mlbId'].iloc[0]}")
            st.text(f"Handedness: {handedness2}")
            with st.expander("Swing Info"):
                    st.write(f"Time of Hit: {time_hit_2:.2f}s")
                    st.write(f"Exit Velocity: {exit_velocity_2:.2f} mph")
                    st.write(f"Hit Spin: {hit_spin2:.2f} rpm")
                    st.write(f"Swing Length: {swing_length_2:.2f} ft")
                    st.write(f"Launch Angle: {launch_angle_2:.2f} degrees")
                    if launch_angle_2 < 10:
                        result = 'ground ball'
                    elif 10 <= launch_angle_2 < 25:
                        result = 'line drive'
                    elif 25 <= launch_angle_2 < 50:
                        result = 'fly ball'
                    else:
                        result = 'pop up'

                    st.write(f"The ball was a {result}")
        plot_df1 = pd.DataFrame(head_positions, columns=['X', 'Y', 'Z'])
        plot_df1['time'] = times

        plot_df2 = pd.DataFrame(handle_positions, columns=['X', 'Y', 'Z'])
        plot_df2['time'] = times

        plot_df3 = pd.DataFrame(head_positions2, columns=['X', 'Y', 'Z'])
        plot_df3['time'] = times2

        plot_df4 = pd.DataFrame(handle_positions2, columns=['X', 'Y', 'Z'])
        plot_df4['time'] = times2
        
        bat1 = plot_df1.merge(plot_df2[['time', 'X', 'Y', 'Z']], on = 'time', how='left')
        bat1.rename(columns={'X_x': 'x_head', 'Y_x': 'y_head', 'Z_x': 'z_head', 'X_y': 'x_handle','Y_y': 'y_handle', 'Z_y': 'z_handle'}, inplace=True)
        bat1 = bat1[(bat1['time'] > 0.15) & (bat1['time'] < 0.7)]
        bat1 = bat1.reset_index(drop=True)
    

        bat2 = plot_df3.merge(plot_df4[['time', 'X', 'Y', 'Z']], on = 'time', how='left')
        bat2.rename(columns={'X_x': 'x_head', 'Y_x': 'y_head', 'Z_x': 'z_head', 'X_y': 'x_handle','Y_y': 'y_handle', 'Z_y': 'z_handle'}, inplace=True)
        bat2 = bat2[(bat2['time'] > 0.15) & (bat2['time'] < 0.7)]
        bat2 = bat2.reset_index(drop=True)

        df1 = bat1
        df2 = bat2

        def calculate_bat_metrics(df):
            df['dx'] = df['x_head'] - df['x_handle']
            df['dy'] = df['y_head'] - df['y_handle']
            df['dz'] = df['z_head'] - df['z_handle']

            df['bat_length'] = np.sqrt(df['dx']**2 + df['dy']**2 + df['dz']**2)
            df['dx_norm'] = df['dx'] / df['bat_length']
            df['dy_norm'] = df['dy'] / df['bat_length']
            df['dz_norm'] = df['dz'] / df['bat_length']

            df['x_sweet_spot'] = df['x_head'] - 0.5 * df['dx_norm']
            df['y_sweet_spot'] = df['y_head'] - 0.5 * df['dy_norm']
            df['z_sweet_spot'] = df['z_head'] - 0.5 * df['dz_norm']

            df['vx'] = df['x_sweet_spot'].diff() / df['time'].diff()
            df['vy'] = df['y_sweet_spot'].diff() / df['time'].diff()
            df['vz'] = df['z_sweet_spot'].diff() / df['time'].diff()
            df = df.dropna().reset_index(drop=True)

            df['v_resultant'] = np.sqrt(df['vx']**2 + df['vy']**2 + df['vz']**2)

            df['VAA_rad'] = np.arctan2(df['vz'], np.sqrt(df['vx']**2 + df['vy']**2))
            df['VAA_deg'] = np.degrees(df['VAA_rad'])

            df['perp_x'] = -df['dz_norm']
            df['perp_z'] = df['dx_norm']

            df['HAA_rad'] = np.arctan2(df['perp_x'], df['perp_z'])
            df['HAA_deg'] = np.degrees(df['HAA_rad'])

            df['horizontal_distance'] = np.sqrt(df['dx']**2 + df['dy']**2)
            df['bat_angle'] = np.arctan2(df['dz'], df['horizontal_distance']) * (180 / np.pi)
            df['head_distance'] = np.sqrt(
                (df['x_head'].diff())**2 + 
                (df['y_head'].diff())**2 + 
                (df['z_head'].diff())**2
            )
            df['time_diff'] = df['time'].diff()
            df['head_speed'] = df['head_distance'] / df['time_diff'] * 0.681818

            df = df.dropna().reset_index(drop=True)

            return df

  
        df1 = calculate_bat_metrics(df1)
        df2 = calculate_bat_metrics(df2)

        # Extract the lowest point for df1
        lowest_point_idx1 = df1['z_head'].idxmin()
        lowest_point1 = df1.loc[lowest_point_idx1]

        # Extract the lowest point for df2
        lowest_point_idx2 = df2['z_head'].idxmin()
        lowest_point2 = df2.loc[lowest_point_idx2]

        # # Didn't end up using
        # if handedness2 == "Left-handed":
        #     widest_point_idx2 = df2['x_head'].idxmin()
        #     widest_point2 = df2.loc[widest_point_idx2]
        # else:
        #     widest_point_idx2 = df2['x_head'].idxmax()
        #     widest_point2 = df2.loc[widest_point_idx2]

        # if handedness1 == "Left-handed":
        #     widest_point_idx1 = df1['x_head'].idxmin()
        #     widest_point1 = df1.loc[widest_point_idx1]
        # else:
        #     widest_point_idx1 = df1['x_head'].idxmax()
        #     widest_point1 = df1.loc[widest_point_idx2]

        # Plot the two swing paths
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=df1['x_head'], 
            y=df1['y_head'], 
            z=df1['z_head'], 
            mode='markers',
            marker=dict(size=2, color='blue'),
            name='Player 1 Bat Head Positions',
            hoverinfo='text',
            text=df1['time']
        ))

        fig.add_trace(go.Scatter3d(
            x=df1['x_handle'], 
            y=df1['y_handle'], 
            z=df1['z_handle'], 
            mode='markers',
            marker=dict(size=2, color='blue'),
            name='Player 1 Bat Handle Positions',
            hoverinfo='text',
            text=df1['time']
        ))

        fig.add_trace(go.Scatter3d(
            x=df2['x_head'], 
            y=df2['y_head'], 
            z=df2['z_head'], 
            mode='markers',
            marker=dict(size=2, color='orange'),
            name='Player 2 Bat Head Positions',
            hoverinfo='text',
            text=df2['time']
        ))

        fig.add_trace(go.Scatter3d(
            x=df2['x_handle'], 
            y=df2['y_handle'], 
            z=df2['z_handle'], 
            mode='markers',
            marker=dict(size=2, color='orange'),
            name='Player 2 Bat Handle Positions',
            hoverinfo='text',
            text=df2['time']
        ))

        fig.add_trace(go.Scatter3d(
            x=df1['x_head'], 
            y=df1['y_head'], 
            z=df1['z_head'], 
            mode='markers',
            marker=dict(size=2, color='blue'),
            name='Player 1 Bat Head Positions',
            hoverinfo='text',
            text=df1['time']
        ))

        fig.add_trace(go.Scatter3d(
            x=df1['x_handle'], 
            y=df1['y_handle'], 
            z=df1['z_handle'], 
            mode='markers',
            marker=dict(size=2, color='blue'),
            name='Player 1 Bat Handle Positions',
            hoverinfo='text',
            text=df1['time']
        ))
        frames = []

        last_known_head_speed1 = None
        last_known_attack_angle1 = None
        last_known_real_attack_angle1 = None
        last_known_real_attack_angle2 = None
        last_known_head_speed2 = None
        last_known_attack_angle2 = None

        for t in sorted(set(df1['time']).union(set(df2['time']))):
            lines_x1, lines_y1, lines_z1 = [], [], []
            lines_x2, lines_y2, lines_z2 = [], [], []

            for i in range(len(df1)):
                if df1['time'].iloc[i] <= t:
                    lines_x1.extend([df1.iloc[i]['x_head'], df1.iloc[i]['x_handle'], None])
                    lines_y1.extend([df1.iloc[i]['y_head'], df1.iloc[i]['y_handle'], None])
                    lines_z1.extend([df1.iloc[i]['z_head'], df1.iloc[i]['z_handle'], None])
                    last_known_head_speed1 = df1['head_speed'].iloc[i]
                    last_known_attack_angle1 = df1['bat_angle'].iloc[i]
                    last_known_real_attack_angle1 = df1['VAA_deg'].iloc[i]

            for i in range(len(df2)):
                if df2['time'].iloc[i] <= t:
                    lines_x2.extend([df2.iloc[i]['x_head'], df2.iloc[i]['x_handle'], None])
                    lines_y2.extend([df2.iloc[i]['y_head'], df2.iloc[i]['y_handle'], None])
                    lines_z2.extend([df2.iloc[i]['z_head'], df2.iloc[i]['z_handle'], None])
                    last_known_head_speed2 = df2['head_speed'].iloc[i]
                    last_known_attack_angle2 = df2['bat_angle'].iloc[i]
                    last_known_real_attack_angle2 = df2['VAA_deg'].iloc[i]
            
            frame_data = [
                go.Scatter3d(
                    x=lines_x1,
                    y=lines_y1,
                    z=lines_z1,
                    mode='lines',
                    line=dict(width=2, color='blue'),
                    showlegend=False
                ),
                go.Scatter3d(
                    x=lines_x2,
                    y=lines_y2,
                    z=lines_z2,
                    mode='lines',
                    line=dict(width=2, color='orange'),
                    showlegend=False
                )
            ]

            annotations = []
            if last_known_head_speed1 is not None and last_known_attack_angle1 is not None and last_known_real_attack_angle1 is not None:
                annotations.append(go.layout.Annotation(
                    text=f"Player 1 Bat Speed: {last_known_head_speed1:.2f} mph<br>Player 1 Bat Angle: {last_known_attack_angle1:.2f} degrees<br>Player 1 Attack Angle: {last_known_real_attack_angle1:.2f} degrees",
                    x=1.2,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12)
                ))
            
            if last_known_head_speed2 is not None and last_known_attack_angle2 is not None and last_known_real_attack_angle1 is not None:
                annotations.append(go.layout.Annotation(
                    text=f"Player 2 Bat Speed: {last_known_head_speed2:.2f} mph<br>Player 2 Bat Angle: {last_known_attack_angle2:.2f} degrees<br>Player 2 Attack Angle: {last_known_real_attack_angle2:.2f} degrees",
                    x=1.1,
                    y=0.4,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=12)
                ))

            frames.append(go.Frame(data=frame_data, name=str(t), layout=go.Layout(annotations=annotations)))

        fig.update(frames=frames)

        fig.add_trace(go.Scatter3d(
            x=[lowest_point1['x_head']],
            y=[lowest_point1['y_head']],
            z=[lowest_point1['z_head']],
            mode='markers+text',
            marker=dict(size=5, color='purple'),
            text=[f"Lowest Point"],
            textposition="bottom center",
            showlegend=False
        ))

        fig.add_trace(go.Scatter3d(
            x=[lowest_point2['x_head']],
            y=[lowest_point2['y_head']],
            z=[lowest_point2['z_head']],
            mode='markers+text',
            marker=dict(size=5, color='yellow'),
            text=[f"Lowest Point"],
            textposition="bottom center",
            showlegend=False
        ))

        time_hit_index = df1.index[df1['time'] == time_hit_1].tolist()[0]
        fig.add_trace(go.Scatter3d(
            x=[df1['x_head'][time_hit_index], df1['x_handle'][time_hit_index]],
            y=[df1['y_head'][time_hit_index], df1['y_handle'][time_hit_index]],
            z=[df1['z_head'][time_hit_index], df1['z_handle'][time_hit_index]],
            mode='lines',
            line=dict(color='purple', width=4),
            name='Player 1: Hit'
        ))

        fig.add_trace(go.Scatter3d(
            x=[(df1['x_head'][time_hit_index] + df1['x_handle'][time_hit_index]) / 2],
            y=[(df1['y_head'][time_hit_index] + df1['y_handle'][time_hit_index]) / 2],
            z=[(df1['z_head'][time_hit_index] + df1['z_handle'][time_hit_index]) / 2],
            mode='text',
            textposition="top center",
            showlegend=False
        ))

        time_hit_index2 = df2.index[df2['time'] == time_hit_2].tolist()[0]
        fig.add_trace(go.Scatter3d(
            x=[df2['x_head'][time_hit_index2], df2['x_handle'][time_hit_index2]],
            y=[df2['y_head'][time_hit_index2], df2['y_handle'][time_hit_index2]],
            z=[df2['z_head'][time_hit_index2], df2['z_handle'][time_hit_index2]],
            mode='lines',
            line=dict(color='yellow', width=4),
            name='Player 2: Hit'
        ))

        fig.add_trace(go.Scatter3d(
            x=[(df2['x_head'][time_hit_index2] + df2['x_handle'][time_hit_index2]) / 2],
            y=[(df2['y_head'][time_hit_index2] + df2['y_handle'][time_hit_index2]) / 2],
            z=[(df2['z_head'][time_hit_index2] + df2['z_handle'][time_hit_index2]) / 2],
            mode='text',
            textposition="top center",
            showlegend=False
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title='X Coordinate',
                yaxis_title='Y Coordinate',
                zaxis_title='Z Coordinate', 
                camera=dict(
                    eye=dict(x=-0, y=1, z=2)  
                )
            ),
            legend_title='Event',
            height=700,
            width=800,
            margin=dict(l=0, r=0, b=10, t=10),
            updatemenus=[{
                'buttons': [
                    {
                        'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}],
                        'label': 'Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                        'label': 'Pause',
                        'method': 'animate'
                    }
                ],
                'direction': 'right',
                'pad': {'r': 10, 't': 87},
                'showactive': False,
                'type': 'buttons',
                'x': 0.3,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }],
            sliders=[{
                'steps': [{'args': [[str(time)], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate'}],
                        'label': str(time), 'method': 'animate'} for time in sorted(set(df1['time']).union(set(df2['time'])))],
                'transition': {'duration': 50},
                'x': 0.3,
                'len': 0.6,
                'currentvalue': {'font': {'size': 20}, 'prefix': 'Time:', 'visible': True, 'xanchor': 'center'},
                'pad': {'b': 10, 't': 30},
                'xanchor': 'left',
                'yanchor': 'top'
            }]
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Additional Information"):
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=df1['time'],
                y=df1['head_speed'],
                mode='lines+markers',
                name='Player 1 Bat Speed',
                line=dict(color='blue'),
                marker=dict(size=6)
            ))
            fig1.add_trace(go.Scatter(
                x=df2['time'],
                y=df2['head_speed'],
                mode='lines+markers',
                name='Player 2 Bat Speed',
                line=dict(color='orange'),
                marker=dict(size=6)
            ))

            fig1.add_trace(go.Scatter(
                x=[time_hit_1],
                y=[df1.loc[df1['time'] == time_hit_1, 'head_speed'].values[0]],
                mode='markers',
                name='Player 1 Time Hit',
                marker=dict(color='purple', size=6)
            ))

            fig1.add_trace(go.Scatter(
                x=[time_hit_2],
                y=[df2.loc[df2['time'] == time_hit_2, 'head_speed'].values[0]],
                mode='markers',
                name='Player 2 Time Hit',
                marker=dict(color='yellow', size=6)
            ))

            fig1.update_layout(
                title='Bat Speed vs Time',
                xaxis_title='Time',
                yaxis_title='Bat Speed (mph)',
                template='plotly_white',
                height=300 
            )

            st.plotly_chart(fig1, use_container_width=True)

        
            fig2 = go.Figure()

            fig2.add_trace(go.Scatter(
                x=df1['time'],
                y=df1['VAA_deg'],
                mode='lines+markers',
                name='Player 1 Vertical Attack Angle',
                line=dict(color='blue'),
                marker=dict(size=6)
            ))

            fig2.add_trace(go.Scatter(
                x=df2['time'],
                y=df2['VAA_deg'],
                mode='lines+markers',
                name='Player 2 Vertical Attack Angle',
                line=dict(color='orange'),
                marker=dict(size=6)
            ))

            fig2.add_trace(go.Scatter(
                x=[time_hit_1],
                y=[df1.loc[df1['time'] == time_hit_1, 'VAA_deg'].values[0]],
                mode='markers',
                name='Player 2 Time Hit',
                marker=dict(color='purple', size=6)
            ))

            fig2.add_trace(go.Scatter(
                x=[time_hit_2],
                y=[df2.loc[df2['time'] == time_hit_2, 'VAA_deg'].values[0]],
                mode='markers',
                name='Player 2 Time Hit',
                marker=dict(color='yellow', size=6)
            ))

            fig2.update_layout(
                title='Vertical Attack Angle vs Time',
                xaxis_title='Time',
                yaxis_title='Attack Angle',
                template='plotly_white',
                height=300  
            )
            st.plotly_chart(fig2, use_container_width=True)

    if __name__ == "__main__":
        df_person1 = df_A[df_A['event_id_updated'] == selected_event_id_A]
        df_person2 = df_B[df_B['event_id_updated'] == selected_event_id_B]
        main(df_person1, df_person2)

# Def'ns
elif page == 'Info':
    st.title('Info')
    st.write('Launch Angle: From the MLB glossary, a burner is when launch angle is under 8 degrees (red), hit under is when launch angle is greater than 40 (red), sweet spot is between 8 to 32 degrees (green) and flare is between 33 and 40 degrees (yellow) ')
    st.write('On Plane: Difference between pitch approach angle and bat attack angle.')
    st.write('Contact from Head of the Bat: Where the contact from the ball was made on the bat, relative to the top of the bat.')
    st.write('Swing: If the attack angle is between 0-4 degrees, the swing is level (yellow). If the attack angle is greater than 4, than the swing is upper cut (green), otherwise under cut (red).')
    st.write('Vertical Attack Angle: The angle between the a perpendicular line at the sweet spot of the bat (0.5 feet from the head of the bat) and the ground.')
    st.write('Vertical Bat Angle: The angle between the bat and an imaginary line on the y-axis.')
    st.write('Horizontal Attack Angle: The angle between the a perpendicular line at the sweet spot of the bat (0.5 feet from the head of the bat) and the z axis.')
    

