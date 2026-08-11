""" Extracts recorded videos from tensorboard events files. """
#!/usr/bin/env python3
import argparse
import io
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator)


def iter_gif_frames(encoded_gif):
  """ Yields rgb arrays from an encoded gif. """
  gif = Image.open(io.BytesIO(encoded_gif))
  try:
    while True:
      yield np.array(gif.convert("RGB"))
      gif.seek(gif.tell() + 1)
  except EOFError:
    pass


def video_tags(event_accumulator):
  """ Returns image tags that contain at least one video. """
  return [tag for tag in event_accumulator.Tags().get("images", [])
          if event_accumulator.Images(tag)]


def extract_videos(logdir, outdir=None, tag=None, fps=15):
  """ Extracts videos from the events file in the given logdir.

  Videos are written to mp4 files (one per recorded step), named after the
  summary tag and the step at which they were recorded.
  """
  event_accumulator = EventAccumulator(
      logdir, size_guidance={"images": 0})
  event_accumulator.Reload()

  tags = video_tags(event_accumulator)
  if not tags:
    raise ValueError(f"no video summaries found in {logdir}")
  if tag is not None:
    if tag not in tags:
      raise ValueError(f"tag {tag!r} not found, available tags: {tags}")
    tags = [tag]

  outdir = outdir or os.path.join(logdir, "videos")
  os.makedirs(outdir, exist_ok=True)
  for tag_name in tags:
    for image_event in event_accumulator.Images(tag_name):
      frames = list(iter_gif_frames(image_event.encoded_image_string))
      fname = os.path.join(
          outdir,
          f"{tag_name.replace('/', '_')}_step{image_event.step}.mp4")
      imageio.mimsave(fname, frames, fps=fps)
      print(f"step={image_event.step}: {len(frames)} frames -> {fname}")


def main():
  """ Entry point of the script. """
  parser = argparse.ArgumentParser(
      description="Extract videos from tensorboard events files.")
  parser.add_argument("--logdir", required=True,
                      help="path to the directory with events files")
  parser.add_argument("--tag", default=None,
                      help="video summary tag to extract (default: all tags)")
  parser.add_argument("--outdir", default=None,
                      help="output directory (default: logdir/videos)")
  parser.add_argument("--fps", type=int, default=15,
                      help="frames per second of output videos (default: 15)")
  args = parser.parse_args()

  extract_videos(args.logdir, outdir=args.outdir, tag=args.tag, fps=args.fps)


if __name__ == "__main__":
  main()
