import ast
import pandas as pd
import numpy as np

def extract_bat_positions(json_str):
    events = ast.literal_eval(json_str)
    head_positions = [event['head']['pos'] for event in events if 'head' in event]
    handle_positions = [event['handle']['pos'] for event in events if 'handle' in event]
    times = [event['time'] for event in events]
    return head_positions, handle_positions, times

def extract_ball_positions(json_str):
    events = ast.literal_eval(json_str)
    positions = [event['pos'] for event in events if 'pos' in event]
    times = [event['time'] for event in events]
    return positions, times


def calculate_averages(df_person):
    results = []
    for index, row in df_person.iterrows():
        # Extract data from total_bat column
        head_positions, handle_positions, times = extract_bat_positions(row['total_bat'])

        positive_indices = [i for i, t in enumerate(times) if t >= 0]
        head_positions = [head_positions[i] for i in positive_indices]
        handle_positions = [handle_positions[i] for i in positive_indices]
        times = [times[i] for i in positive_indices]

        # Create DataFrames for head and handle positions
        plot_df2 = pd.DataFrame(head_positions, columns=['X', 'Y', 'Z'])
        plot_df2['time'] = times

        plot_df3 = pd.DataFrame(handle_positions, columns=['X', 'Y', 'Z'])
        plot_df3['time'] = times

        # Extract data from total_ball column
        positions, times = extract_ball_positions(row['total_ball'])

        plot_df4 = pd.DataFrame(positions, columns=['X', 'Y', 'Z'])
        plot_df4['time'] = times

        # Just a bunch of variables
        time_hit = row['time_hit']

        # Ball trajectory before it was hit
        plot_df4 = plot_df4[plot_df4['time'] <= time_hit]
        # Head of bat at contact time
        plot_df22 = plot_df2[plot_df2['time'] == time_hit]

        plot_df4['dX'] = plot_df4['X'].diff()
        plot_df4['dY'] = plot_df4['Y'].diff()
        plot_df4['dZ'] = plot_df4['Z'].diff()
        last_dx = plot_df4['dX'].iloc[-1]
        last_dy = plot_df4['dY'].iloc[-1]
        last_dz = plot_df4['dZ'].iloc[-1]

        last_horizontal_displacement = np.sqrt(last_dx**2 + last_dy**2)

        # Finding the angle of the incoming pitch 
        last_approach_angle = np.degrees(np.arctan(last_dz / last_horizontal_displacement))
        # Merge data to get one dataframe with all the bat's positional coordinates
        bat = plot_df2.merge(plot_df3[['time', 'X', 'Y', 'Z']], on='time', how='left')
        bat.rename(columns={'X_x': 'x_head', 'Y_x': 'y_head', 'Z_x': 'z_head', 'X_y': 'x_handle', 'Y_y': 'y_handle', 'Z_y': 'z_handle'}, inplace=True)

        # Calculate the differences
        bat['dx'] = bat['x_head'] - bat['x_handle']
        bat['dy'] = bat['y_head'] - bat['y_handle']
        bat['dz'] = bat['z_head'] - bat['z_handle']

        # Find the length of the bat
        bat['bat_length'] = np.sqrt(bat['dx']**2 + bat['dy']**2 + bat['dz']**2)

        bat['dx_norm'] = bat['dx'] / bat['bat_length']
        bat['dy_norm'] = bat['dy'] / bat['bat_length']
        bat['dz_norm'] = bat['dz'] / bat['bat_length']

        bat['x_sweet_spot'] = bat['x_head'] - 0.5 * bat['dx_norm']
        bat['y_sweet_spot'] = bat['y_head'] - 0.5 * bat['dy_norm']
        bat['z_sweet_spot'] = bat['z_head'] - 0.5 * bat['dz_norm']

        bat['vx'] = bat['x_sweet_spot'].diff() / bat['time'].diff()
        bat['vy'] = bat['y_sweet_spot'].diff() / bat['time'].diff()
        bat['vz'] = bat['z_sweet_spot'].diff() / bat['time'].diff()

        bat = bat.dropna().reset_index(drop=True)
        bat['v_resultant'] = np.sqrt(bat['vx']**2 + bat['vy']**2 + bat['vz']**2)

        # Calculating the Vertical Attack Angle
        bat['VAA_rad'] = np.arctan2(bat['vz'], np.sqrt(bat['vx']**2 + bat['vy']**2))
        bat['VAA_deg'] = np.degrees(bat['VAA_rad'])

        attack_angle = bat[bat['time'] == time_hit]
        vaa = attack_angle['VAA_deg'].iloc[0]

        # Calculate the Horizontal Attack Angle
        bat['perp_x'] = -bat['dz_norm']
        bat['perp_z'] = bat['dx_norm']

        bat['HAA_rad'] = np.arctan2(bat['perp_x'], bat['perp_z'])
        bat['HAA_deg'] = np.degrees(bat['HAA_rad'])
        haaat = bat[bat['time'] == time_hit]
        haa = haaat['HAA_deg'].iloc[0]

        # Calculate the bat speed
        bat['head_distance'] = np.sqrt(bat[['x_head', 'y_head', 'z_head']].diff().pow(2).sum(axis=1))
        bat['time_diff'] = bat['time'].diff()
        bat['head_speed'] = bat['head_distance'] / bat['time_diff'] * 0.681818  # conversion factor for ft/s to mph

        bat = bat.dropna().reset_index(drop=True)
        bat = bat[bat['time'] == time_hit]
        bat_speed_at_time_hit = bat['head_speed'].iloc[0] if not bat.empty else np.nan

        results.append({
            'bat_speed': bat_speed_at_time_hit,
            'VAA_deg': vaa,
            'HAA_deg': haa,
            'Pitch': last_approach_angle
        })

    # Calculate averages for the selected person
    results_df = pd.DataFrame(results)
    average_bat_speed = results_df['bat_speed'].mean()
    average_VAA_deg = results_df['VAA_deg'].mean()
    average_HAA_deg = results_df['HAA_deg'].mean()
    average_pitch = results_df['Pitch'].mean()

    return average_bat_speed, average_VAA_deg, average_HAA_deg, average_pitch
