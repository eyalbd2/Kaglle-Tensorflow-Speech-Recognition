import plotly.offline as py
# import plotly.plotly as plt
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import matplotlib.pyplot as plt




# Calculate
number_of_recordings = [2377, 2375, 2375, 2359, 2353, 2367, 2367, 2357, 2380, 2372, 43414]
dirs = ['Yes', 'No', 'Up', 'Down', 'Left', 'Right', 'On', 'Off', 'Stop', 'Go', 'Unknown']
dirs_num = [0, 1 , 2, 3, 4, 5, 6, 7, 8, 9, 10]

Clesses = []
for i in range(11):
    temp_len = number_of_recordings[i]
    temp_class = i
    for j in range(temp_len):
        Clesses.append(temp_class)

# # Plot
# data = [go.Histogram(x=dirs, y=number_of_recordings)]
# trace = go.Bar(
#     x=dirs,
#     y=number_of_recordings,
#     marker=dict(color = number_of_recordings, colorscale='Viridius', showscale=True
#     ),
# )
# layout = go.Layout(
#     title='Number of recordings in given label',
#     xaxis = dict(title='Words'),
#     yaxis = dict(title='Number of recordings')
# )
# # py.iplot(go.Figure(data=[trace], layout=layout))
# plt.iplot(go.Figure(data=[trace], layout=layout), filename = 'basic-line')

# # show labels probabillity in train dataset
plt.hist(Clesses, bins = dirs_num)
plt.title("histogram of Train set")
plt.show()
