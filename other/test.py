import plotly.io as pio
img = fig.to_image(format="pdf", engine="kaleido")   # returns bytes
print(len(img))     # should be > 0