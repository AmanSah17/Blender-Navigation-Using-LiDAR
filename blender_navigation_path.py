#!/usr/bin/env python
# coding: utf-8

# In[72]:


import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from mpl_toolkits.mplot3d import Axes3D


# In[92]:


door_3d = pd.read_csv("C:\\Users\\amans\\Downloads\\door_3d.csv")
plane_wall = pd.read_csv("C:\\Users\\amans\\Downloads\\plane_wall_coordinates.csv")


#Create 3D scatter plot using Plotly Express
fig = px.scatter_3d(door_3d, x='X', y='Y', z='Z', color='Y', symbol='Y', size_max=10, opacity=0.7, labels={'X': 'X Label', 'Y': 'Thickness (Y Label)', 'Z': 'Z Label'})
fig.update_traces(marker=dict(size=10), selector=dict(mode='markers'))

# Add points from plane_wall
fig.add_trace(px.scatter_3d(plane_wall, x='X', y='Y', z='Z', color='Y', symbol='Y', size_max=10, opacity=0.7, labels={'X': 'X Label', 'Y': 'Thickness (Y Label)', 'Z': 'Z Label'}).update_traces(marker=dict(size=5)).data[0])

# Show the plot
fig.show()


# In[ ]:





# In[ ]:





# In[74]:


# Plotting the 3D map
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting points from door_data as lines
ax.plot(door_3d['X'], door_3d['Y'], door_3d['Z'], label='Door 3d', marker='o')

# Plotting points from plane_wall as lines
ax.plot(plane_wall['X'], plane_wall['Y'], plane_wall['Z'], label='Plane Wall', marker='o')

# Set axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Thickness (Y Label)')
ax.set_zlabel('Z Label')

# Add a legend
ax.legend()

# Show the plot
plt.show()


# In[75]:


# Plotting the 3D map
fig = plt.figure(figsize=(10, 8))  # Increase the figure size
ax = fig.add_subplot(111, projection='3d')

# Plotting points from door_data as lines with larger markers
ax.plot(door_3d['X'], door_3d['Y'], door_3d['Z'], label='Door Data', marker='o', markersize=0.2)

# Plotting points from plane_wall as lines with larger markers
ax.plot(plane_wall['X'], plane_wall['Y'], plane_wall['Z'], label='Plane Wall', marker='o', markersize=10)

# Set axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Thickness (Y Label)')
ax.set_zlabel('Z Label')


ax.set_box_aspect([1, 1, 1])  # # Adjust aspect ratio for a more accurate representation of the data

# Add a legend
ax.legend()

# Show the plot
plt.show()


# In[ ]:





# In[93]:


df = pd.read_csv("C:\\Users\\amans\\Downloads\\plane_wall_coordinates.csv")


# In[94]:


import plotly.graph_objects as go
from scipy.interpolate import interp1d
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
fig = px.scatter_3d(df, x='X', y='Y', z='Z', color='Z', symbol='Z', size_max=10,
                    title='3D Scatter Plot For Wall Data', labels={'X': 'X Label', 'Y': 'Y Label', 'Z': 'Z Label'})

fig.show()


# In[95]:


# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points using DataFrame columns
ax.scatter(df['X'], df['Y'], df['Z'], c='r', marker='o')

# Set labels for the axes
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Show the plot
plt.show()


# In[96]:


plt.figure(figsize=(12, 8))
sns.set_palette("viridis")  # Set color palette
sns.countplot(data=df, x='X', hue='Z')


plt.xlabel('X Value')
plt.ylabel('Count of Z Values')

plt.legend(title='Z Value')
plt.title('Bar Plot of Z for Each X')
plt.show()


# # DATA VISUALIZATION FOR SPECIFIC RANGE

# In[97]:


import plotly.graph_objects as go



# Forcibly set the 0th index 'X' value to -50(to visualize your range of values)
df.at[0, 'X'] = -50

# Filter negative values of X
neg_x_df = df[df['X'] < 0]


fig = go.Figure()

# Add scatter plot for negative X
scatter = fig.add_trace(
    go.Scatter3d(
        x=neg_x_df['X'],
        y=neg_x_df['Y'],
        z=neg_x_df['Z'],
        mode='markers',
        marker=dict(size=10, color='blue'),
        name='Scatter Points (Negative X)'
    )
)

# Add red line joining the points
line = fig.add_trace(
    go.Scatter3d(
        x=neg_x_df['X'],
        y=neg_x_df['Y'],
        z=neg_x_df['Z'],
        mode='lines',
        line=dict(color='red', width=4),
        name='Line (Negative X)'
    )
)


fig.update_layout(scene=dict(aspectmode='cube'))
fig.update_layout(scene=dict(xaxis_title='X Value', yaxis_title='Y Value', zaxis_title='Z Value'))
fig.show()


# In[98]:


# Forcibly set the 0th index 'X' value to -50
df.at[0, 'X'] = -50
fig = go.Figure()

# Loop through each 'X' value
for x_value in df['X'].unique():
    x_subset = df[df['X'] == x_value]

    # Add scatter plot for current X
    scatter = fig.add_trace(
        go.Scatter3d(
            x=x_subset['X'],
            y=x_subset['Y'],
            z=x_subset['Z'],
            mode='markers',
            marker=dict(size=10, color='blue'),
            name=f'Scatter Points (X={x_value})'
        )
    )

    # Add line joining the points for current X
    line = fig.add_trace(
        go.Scatter3d(
            x=x_subset['X'],
            y=x_subset['Y'],
            z=x_subset['Z'],
            mode='lines',
            line=dict(color='red', width=4),
            name=f'Line (X={x_value})'
        )
    )


fig.update_layout(scene=dict(aspectmode='cube'))
fig.update_layout(scene=dict(xaxis_title='X Value', yaxis_title='Y Value', zaxis_title='Z Value'))
fig.show()


# # DATA VISUALIZATION : TRYING TO FIND NEXT CO-RRELATAED POINTS IN THE POSSIBLE DOMAIN (ACCORDING TO CONSTRAINT) 

# In[99]:


import pandas as pd
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import interp1d

# Forcibly set the 0th index 'X' value to -50
df.at[0, 'X'] = -50

# Create a 3D scatter plot
fig = go.Figure()

# Loop through each 'X' value
for x_value in df['X'].unique():
    x_subset = df[df['X'] == x_value]

    # Add scatter plot for current X
    scatter = fig.add_trace(
        go.Scatter3d(
            x=x_subset['X'],
            y=x_subset['Y'],
            z=x_subset['Z'],
            mode='markers',
            marker=dict(size=10, color='blue'),
            name=f'Scatter Points (X={x_value})'
        )
    )

    # Add line joining the points for current X
    line = fig.add_trace(
        go.Scatter3d(
            x=x_subset['X'],
            y=x_subset['Y'],
            z=x_subset['Z'],
            mode='lines',
            line=dict(color='blue', width=4),
            name=f'Line (X={x_value})'
        )
    )

# Interpolate Z values for each step jump
z_values = np.arange(100, -5, -5)  # Start from 100 and decrease in steps of 5 until 0

# Create interpolated Z values for each X
for x_value in df['X'].unique():
    x_subset = df[df['X'] == x_value]
    interpolated_z = np.interp(np.arange(len(x_subset)), [0, len(x_subset) - 1], [x_subset['Z'].iloc[0], x_subset['Z'].iloc[-1]])

    # Add interpolated Z values to the plot
    fig.add_trace(
        go.Scatter3d(
            x=x_subset['X'],
            y=x_subset['Y'],
            z=interpolated_z,
            mode='markers+lines',
            marker=dict(size=10, color='blue'),
            line=dict(color='blue', width=4),
            name=f'Interpolated Line (X={x_value})'
        )
    )

# Add text labels for each point
for i, row in df.iterrows():
    fig.add_trace(
        go.Scatter3d(
            x=[row['X']],
            y=[row['Y']],
            z=[row['Z']],
            mode='text',
            text=[f'Target{i + 1}'],
            textposition='top center',
            textfont=dict(size=10),
            showlegend=False
        )
    )

fig.update_layout(scene=dict(aspectmode='cube'))
fig.update_layout(scene=dict(xaxis_title='X Value', yaxis_title='Y Value', zaxis_title='Z Value'))
fig.show()


# In[ ]:





# # CREATE WAY POINTS TO NAVIGATE THROUGH THE COLLECTED CO-ORDINATES

# In[100]:


import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("C:\\Users\\amans\\Downloads\\plane_wall_coordinates.csv")

# Forcibly set the 0th index 'X' value to -50
df.at[0, 'X'] = -50

# Create a 3D scatter plot
fig = go.Figure()

# Add scatter plot
scatter = fig.add_trace(
    go.Scatter3d(
        x=df['X'],
        y=df['Y'],
        z=df['Z'],
        mode='markers',
        marker=dict(size=2, color='blue'),
        name='Scatter Points'
    )
)

# Add arrows
arrows = []
for i in range(len(df) - 1):
    arrow = go.Cone(
        x=[df['X'][i]],
        y=[df['Y'][i]],
        z=[df['Z'][i]],
        u=[df['X'][i + 1] - df['X'][i]],
        v=[df['Y'][i + 1] - df['Y'][i]],
        w=[df['Z'][i + 1] - df['Z'][i]],
        colorscale='greens',
        sizemode='scaled',
        sizeref=0.25, 
        showscale=False
    )
    arrows.append(arrow)

fig.add_traces(arrows)

# Loop through each 'X' value
for x_value in df['X'].unique():
    x_subset = df[df['X'] == x_value]

    # Add scatter plot for current X
    scatter = fig.add_trace(
        go.Scatter3d(
            x=x_subset['X'],
            y=x_subset['Y'],
            z=x_subset['Z'],
            mode='markers',
            marker=dict(size=1, color='red'),
            name=f'Scatter Points (X={x_value})'
        )
    )

    # Add line joining the points for current X
    line = fig.add_trace(
        go.Scatter3d(
            x=x_subset['X'],
            y=x_subset['Y'],
            z=x_subset['Z'],
            mode='lines',
            line=dict(color='red', width=4),
            name=f'Line (X={x_value})'
        )
    )

fig.update_layout(scene=dict(aspectmode='cube'))
fig.update_layout(scene=dict(xaxis_title='X Value', yaxis_title='Y Value', zaxis_title='Z Value'))
fig.show()


# In[101]:


fig = px.scatter_3d(door_3d, x='X', y='Y', z='Z', color='Y', symbol='Y', size_max=10,
                    opacity=0.7, labels={'X': 'X Label', 'Y': 'Thickness (Y Label)', 'Z': 'Z Label'})
fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
fig.add_trace(px.scatter_3d(plane_wall, x='X', y='Y', z='Z', color='Y', symbol='Y',
                            size_max=10, opacity=0.7,
                            labels={'X': 'X Label', 'Y': 'Thickness (Y Label)', 'Z': 'Z Label'}).update_traces(marker=dict(size=5)).data[0])
fig.update_layout(scene=dict(zaxis=dict(title=None)))
fig.show()


# # 3D VISUALIZATION Plot of X vs Z for Door and Wall

# In[102]:


# Create 2D scatter plot using Plotly Express
fig = px.scatter()
fig.add_trace(px.scatter(door_3d, x='X', y='Z', color='Y',
                         labels={'X': 'X Label', 'Y': 'Thickness (Y Label)', 'Z': 'Z Label'}).data[0])
fig.add_trace(px.scatter(plane_wall, x='X', y='Z', color='Y', 
                         labels={'X': 'X Label', 'Y': 'Thickness (Y Label)', 'Z': 'Z Label'}).data[0])


fig.update_layout(title='Scatter Plot of X vs Z for Door and Wall',
                  xaxis_title='X',
                  yaxis_title='Z')
fig.show()


# In[103]:


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation



df.at[0, 'X'] = -50
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['X'], df['Y'], df['Z'], s=50, c='blue',
                     edgecolors='black', alpha=0.7, label='Scatter Points')

# Set labels for the axes
ax.set_xlabel('X Value')
ax.set_ylabel('Y Value')
ax.set_zlabel('Z Value')

# Add legend with color titles
legend1 = ax.legend(handles=[scatter], labels=['Scatter Points'])
ax.add_artist(legend1)

# Animation function
def update(frame):
    ax.clear()
    scatter = ax.scatter(df['X'][:frame], df['Y'][:frame], df['Z'][:frame], s=50, c='blue', edgecolors='black', alpha=0.7, label='Scatter Points')
    ax.set_xlabel('X Value')
    ax.set_ylabel('Y Value')
    ax.set_zlabel('Z Value')
    ax.legend(handles=[scatter], labels=['Scatter Points'])

ani = FuncAnimation(fig, update, frames=len(df), interval=500, repeat=False)
plt.title('Visualization Animated 3D Scatter Plot')
plt.grid(True)
plt.show()


# # HISTOGRAM PLOT FOR DESIGNED WAY PATH

# In[104]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

hist, xedges, yedges = np.histogram2d(df['X'], df['Y'], bins=(20, 20))

xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

dx = dy = 0.5
dz = hist.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, cmap='viridis')

# Set labels for the axes
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Show the plot
plt.title('3D Histogram')
plt.show()



# In[105]:


plt.figure(figsize=(12, 8))
sns.set_palette("viridis")  # Set color palette
sns.countplot(data=df, x='X', hue='Z')

# Set labels for the axes
plt.xlabel('X Value')
plt.ylabel('Count of Z Values')

# Add a legend
plt.legend(title='Z Value')

# Show the plot
plt.title('Bar Plot of Z for Each X')
plt.show()


# In[106]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting points from door_data as lines
ax.plot(door_3d['X'], door_3d['Y'], door_3d['Z'], label='Door 3d', marker='o')

# Plotting points from plane_wall as lines
ax.plot(plane_wall['X'], plane_wall['Y'], plane_wall['Z'], label='Plane Wall', marker='o')

# Set axis labels
ax.set_xlabel('X Label')
ax.set_ylabel('Thickness (Y Label)')
ax.set_zlabel('Z Label')
ax.legend()
plt.show()


# In[107]:


#setting the initial drone -co-ordindates
df.at[0, 'X'] = -50

# Create a scatter plot of X vs Z with larger blue points
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['X'], df['Z'], s=50, c='blue', edgecolors='black', alpha=0.7, label='Scatter Points')

# Annotate each point with its number
for i, txt in enumerate(df.index):
    plt.annotate(txt, (df['X'][i], df['Z'][i]), textcoords="offset points", xytext=(0,5), ha='center')

# Create a red thick line connecting the points
plt.plot(df['X'], df['Z'], color='red', linewidth=2, label='Line')

# Set labels for the axes
plt.xlabel('X Value')
plt.ylabel('Z Value')
legend1 = plt.legend(handles=[scatter], labels=['Scatter Points'])
legend2 = plt.legend(['Line'], loc='upper right')
plt.gca().add_artist(legend1)


plt.title('Scatter Plot of X vs Z with Line')
plt.grid(True)
plt.show()


# # ENTIRE NAVIGATION PATH TO VISIT THE WALL CO-ORDINATES ONLY

# In[108]:


df.at[0, 'X'] = -50

# Create a scatter plot of X vs Z with larger blue points
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['X'], df['Z'], s=50, c='blue', edgecolors='black', alpha=0.7, label='Scatter Points')

for i, txt in enumerate(df.index):
    plt.annotate(txt, (df['X'][i], df['Z'][i]), textcoords="offset points", xytext=(0,5), ha='center')

plt.plot(df['X'], df['Z'], color='red', linewidth=2, label='Line')

for i in range(len(df) - 1):
    plt.arrow(df['X'][i], df['Z'][i], df['X'][i + 1] - df['X'][i], df['Z'][i + 1] - df['Z'][i],
              color='green', shape='full', lw=0, length_includes_head=True, head_width=2)

plt.xlabel('X Value')
plt.ylabel('Z Value')

# Add legend with color titles
legend1 = plt.legend(handles=[scatter], labels=['Scatter Points'])
legend2 = plt.legend(['Line'], loc='upper right')
plt.gca().add_artist(legend1)
plt.title('ENTIRE NAVIGATION Plot of X vs Z with Line and Arrows')
plt.grid(True)
plt.show()


# # ACCORDING TO THE ABOVE CO-ORDINATES GENERATED .BELOW CODES MOVE THE OBJECT TO THOSE SPECIFIC CO-ORDINATES ONLY.
# TO RUN THE CODE OPEN BLENDER > SCRIPT > COPY AND PASTE THE BELOW CODE IN THE PANEL > SELECT AN OBJECT , USING ADD TOOL > MESH > 
# CUBE > LEFT CLICK SELECT THE CUBE > RUN THE CODE ON THE SCRIPT PANEL
import bpy
import os

def create_and_link_trace_marker():
    trace_marker = bpy.data.objects.new("TraceMarker", None)
    bpy.context.scene.collection.objects.link(trace_marker)
    return trace_marker

def set_keyframes(selected_object, trace_marker, coordinates):
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 1  # Initialize frame_end

    for coord in coordinates:
        selected_object.location = coord["location"]
        selected_object.keyframe_insert(data_path="location", index=-1, frame=coord["frame"])

        trace_marker.location = coord["location"]
        trace_marker.keyframe_insert(data_path="location", index=-1, frame=coord["frame"])

        bpy.context.scene.frame_end = coord["frame"]

def set_render_settings(output_filepath):
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    bpy.context.scene.render.filepath = output_filepath
    bpy.context.scene.render.ffmpeg.format = 'MPEG4'
    bpy.context.scene.render.ffmpeg.codec = 'H264'
    bpy.context.scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'

def render_animation():
    bpy.ops.render.render(animation=True)

def main():
    trace_marker = create_and_link_trace_marker()

    if bpy.context.selected_objects:
        selected_object = bpy.context.selected_objects[0]

        # Define the new movement coordinates
        new_movement_coordinates = [
            {"frame": 1, "location": (-50, -15, 0)},
            {"frame": 31, "location": (-50, -15, 100)},
            {"frame": 61, "location": (-45, -15, 100)},
            {"frame": 91, "location": (-45, -15, 0)},
            {"frame": 121, "location": (-40, -15, 0)},
            {"frame": 151, "location": (-40, -15, 100)},
            {"frame": 181, "location": (-35, -15, 100)},
            {"frame": 211, "location": (-35, -15, 0)},
            {"frame": 241, "location": (-30, -15, 0)},
            {"frame": 271, "location": (-30, -15, 100)},
            {"frame": 301, "location": (-25, -15, 100)},
            {"frame": 331, "location": (-25, -15, 0)},
            {"frame": 361, "location": (-20, -15, 0)},
            {"frame": 391, "location": (-20, -15, 100)},
            {"frame": 421, "location": (-20, -15, 100)},
            {"frame": 451, "location": (20, -15, 100)},
            {"frame": 481, "location": (20, -15, 95)},
            {"frame": 511, "location": (-20, -15, 95)},
            {"frame": 541, "location": (-20, -15, 90)},
            {"frame": 571, "location": (20, -15, 90)},
            {"frame": 601, "location": (20, -15, 85)},
            {"frame": 631, "location": (-20, -15, 85)},
            {"frame": 661, "location": (-20, -15, 80)},
            {"frame": 691, "location": (20, -15, 80)},
            {"frame": 721, "location": (20, -15, 100)},
            {"frame": 751, "location": (50, -15, 100)},
            {"frame": 781, "location": (50, -15, 90)},
            {"frame": 811, "location": (20, -15, 90)},
            {"frame": 841, "location": (20, -15, 80)},
            {"frame": 871, "location": (50, -15, 80)},
            {"frame": 901, "location": (50, -15, 70)},
            {"frame": 931, "location": (20, -15, 70)},
            {"frame": 961, "location": (20, -15, 60)},
            {"frame": 991, "location": (50, -15, 60)},
            {"frame": 1021, "location": (50, -15, 50)},
            {"frame": 1051, "location": (20, -15, 50)},
            {"frame": 1081, "location": (20, -15, 30)},
            {"frame": 1111, "location": (50, -15, 30)},
            {"frame": 1141, "location": (50, -15, 20)},
            {"frame": 1171, "location": (20, -15, 20)},
            {"frame": 1201, "location": (20, -15, 10)},
            {"frame": 1231, "location": (50, -15, 10)},
            {"frame": 1261, "location": (50, -15, 5)},
            {"frame": 1291, "location": (20, -15, 5)},
            {"frame": 1321, "location": (20, -15, 0)},
            {"frame": 1351, "location": (50, -15, 0)},
        ]

        set_keyframes(selected_object, trace_marker, new_movement_coordinates)

        for fcurve in selected_object.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'LINEAR'

        output_directory = os.path.expanduser("~/Downloads/")
        output_filepath = os.path.join(output_directory, "way_point.mp4")

        set_render_settings(output_filepath)
        render_animation()

        print(f"Animation saved to: {output_filepath}")

    else:
        print("No object selected")

if __name__ == "__main__":
    main()

# # CHANGE THE CAMERA SETTINGS ACCORDING TO YOUR SPECIFIC LOCATION FROM THE BLENDER > CAMERA > OBJECT > EDIT LOCATION  > ORIENTATION > CHANGE FIELD OF VIEW 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




