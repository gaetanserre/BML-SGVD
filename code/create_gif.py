
import imageio
import os

import re
def sorted_alphanumeric(data):
  convert = lambda text: int(text) if text.isdigit() else text.lower()
  alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
  return sorted(data, key=alphanum_key)

if __name__ == "__main__":
  # List all files in the figures directory
  files = os.listdir("figures")
  # Sort the file alphanumeric
  files = sorted_alphanumeric(files)

  frames = []
  for file in files:
    if file.endswith(".png"):
      frames.append(imageio.imread("figures/" + file))
  
  # Save the frames as a gif
  imageio.mimsave("figures/animation.gif", frames, duration=0.01)
