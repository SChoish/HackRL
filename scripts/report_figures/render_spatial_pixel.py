"""Warm pixel-art observer view of verified spatial traces; no environment imports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap

from PIL import Image, ImageDraw, ImageFont

from report_figures import render_pack_pixel as pack

W, H, SCALE = 960, 540, 2
PAPER, INK, MUTED = "#f4ead8", "#342e28", "#817362"
ACCENTS = ("#4f7691", "#368575", "#b66a45")


def font(size, bold=False):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans" + ("-Bold" if bold else "") + ".ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size=size)


def text(draw, xy, value, size=11, fill=INK, bold=False, anchor=None):
    draw.text(xy, str(value), font=font(size, bold), fill=fill, anchor=anchor)


def fit(draw, xy, value, width, size=14, bold=False, fill=INK):
    while size > 7 and draw.textlength(str(value), font=font(size, bold)) > width:
        size -= 1
    text(draw, xy, value, size, fill, bold)


def snapshot(scene, index):
    n = len(scene["steps"])
    step = scene["steps"][min(index, n) - 1] if index and n else None
    return (step["after"] if step else scene["initial"]), step, index > n


def tile(kind, row=0, col=0, direction=2):
    im = Image.new("RGBA", (42, 42), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if kind == "floor":
        pack._stone_tile(d, 0, 0, row, col)
    elif kind == "wall":
        pack._wall_tile(d, 0, 0, row, col)
        d.line([(1, 1), (40, 1)], fill="#bca283", width=3)
        d.line([(1, 1), (1, 40)], fill="#92775d", width=2)
        d.rectangle((2, 35, 40, 40), fill="#493c31")
    elif kind == "worker":
        pack._worker(d, 0, 0, direction)
    elif kind == "camp":
        pack._delivery(d, 0, 0)
        d.rectangle((6, 5, 8, 24), fill="#59472f")
        d.polygon([(9, 5), (22, 5), (19, 10), (9, 10)], fill="#d9ab4e")
    elif kind == "parcel":
        d.rectangle((10, 26, 33, 32), fill="#9a805f")
        d.rectangle((9, 13, 31, 29), fill="#ac7445", outline="#543b2d", width=2)
        d.polygon([(9, 13), (15, 8), (35, 8), (31, 13)], fill="#d5a967", outline="#543b2d")
        d.polygon([(31, 13), (35, 8), (35, 24), (31, 29)], fill="#815532")
        d.rectangle((18, 13, 22, 29), fill="#f1d9a4")
        d.line([(18, 13), (23, 8)], fill="#f1d9a4", width=3)
    return im


def map_view(canvas, x, scene, geometry, index, accent):
    d = ImageDraw.Draw(canvas)
    floors = {tuple(p) for p in geometry["walkable"]}
    walls = {tuple(p) for p in geometry["walls"]}
    points = floors | walls | {tuple(geometry["camp"]), tuple(geometry["item"])}
    top = min(p[0] for p in points) - 1
    left = min(p[1] for p in points) - 1
    rows = max(p[0] for p in points) - top + 2
    cols = max(p[1] for p in points) - left + 2
    cell = min(28, 252 // max(rows, cols))
    ox, oy = x + (292 - cols * cell) // 2, 136 + (252 - rows * cell) // 2

    def at(p):
        return ox + (int(p[1]) - left) * cell, oy + (int(p[0]) - top) * cell

    def center(p):
        xx, yy = at(p)
        return xx + cell // 2, yy + cell // 2

    def sprite(kind, p, **kwargs):
        im = tile(kind, **kwargs).resize((cell, cell), Image.Resampling.NEAREST)
        canvas.paste(im, at(p), im)

    d.rectangle((ox-4, oy-4, ox+cols*cell+3, oy+rows*cell+3), fill="#433b32")
    for r in range(top, top+rows):
        for c in range(left, left+cols):
            px, py = at((r, c))
            if (r,c) in floors:
                sprite("floor", (r,c), row=r, col=c)
            elif (r,c) in walls:
                sprite("wall", (r,c), row=r, col=c)
            else:
                d.rectangle((px, py, px+cell-1, py+cell-1), fill="#554d40")
                # Texture only: these cells stay outside the walkable set.
                d.rectangle((px+4, py+5, px+7, py+6), fill="#71664f")
                d.rectangle((px+cell-9, py+cell-8, px+cell-6, py+cell-5), fill="#635a48")
    state, step, held = snapshot(scene, index)
    for s in scene["steps"][:index]:
        p0, p1 = s["before"]["player_position"], s["after"]["player_position"]
        if p0 != p1:
            a, b = center(p0), center(p1)
            if s["events"]["crossed_wall"]:
                for k in range(0, 8, 2):
                    u, v = k/8, (k+1)/8
                    d.line([(a[0]+u*(b[0]-a[0]), a[1]+u*(b[1]-a[1])),
                            (a[0]+v*(b[0]-a[0]), a[1]+v*(b[1]-a[1]))], fill=accent, width=3)
            else:
                d.line([a,b], fill=accent, width=2)
    sprite("camp", geometry["camp"])
    if state["item_present"]:
        sprite("parcel", geometry["item"])
    else:
        cx, cy = center(geometry["item"])
        d.rectangle((cx-6,cy-6,cx+6,cy+6), outline="#a28d6c", width=1)
    for name, pos in (("CAMP", geometry["camp"]), ("PARCEL", geometry["item"])):
        cx, cy = center(pos)
        tw = d.textlength(name, font=font(6,True))
        d.rectangle((cx-tw/2-3,cy+13,cx+tw/2+3,cy+22), fill="#433b32")
        text(d, (cx,cy+14), name, 6, "#fff5dd", True, "mt")
    if step and not held:
        e = step["events"]
        if e["crossed_wall"] or e["blocked_dash"]:
            cx, cy = center(e["intermediate_position"])
            color = "#41b69b" if e["crossed_wall"] else "#cf654f"
            d.rectangle((cx-cell//2-1, cy-cell//2-1,cx+cell//2,cy+cell//2), outline=color,width=2)
            if e["blocked_dash"]:
                d.line([(cx-5,cy-5),(cx+5,cy+5)],fill="#ffb8a0",width=3)
                d.line([(cx+5,cy-5),(cx-5,cy+5)],fill="#ffb8a0",width=3)
    sprite("worker", state["player_position"], direction=state["player_direction"])
    cx, cy = center(state["player_position"])
    if state["carrying_item"]:
        d.rectangle((cx+4,cy+1,cx+11,cy+8),fill="#d6a05e",outline="#563a26",width=1)
        d.line([(cx+7,cy+2),(cx+7,cy+7)], fill="#f4d89b",width=2)
    return state, step, held


def event_message(scene, state, step, held):
    if scene["summary"]["success"] and state["delivered_total"]:
        return "DELIVERY COMPLETE", "The parcel is back at camp."
    if held:
        return "EPISODE ENDED", "Goal not reached before " + scene["summary"].get("end_reason", "trace end") + "."
    if step is None:
        return "READY", "Collect the parcel. Return it to camp."
    e = step["events"]
    if e["blocked_dash"]:
        return "DASH BLOCKED", "The wall stops the dash. Position unchanged."
    if e["crossed_wall"]:
        return "WALL PASSED", "One dash crosses the solid wall."
    if e["picked_up"]:
        return "PARCEL COLLECTED", "Carry the parcel back to camp."
    return step["action_name"].replace("_", " "), "Returning to camp." if state["carrying_item"] else "Moving toward the parcel."


def render_frame(trace, index):
    if trace.get("schema_version") != "hackrl_spatial_render_v1":
        raise ValueError("expected a verified spatial render trace")
    scenes = trace["scenes"]
    if not 1 <= len(scenes) <= 3:
        raise ValueError("renderer supports one to three panels")
    im = Image.new("RGB", (W,H), PAPER)
    d = ImageDraw.Draw(im)
    text(d,(24,11),"HACKRL  /  SPATIAL WALL PASS",9,MUTED,True)
    text(d,(24,30),"A shortcut through a solid wall",25,INK,True)
    text(d,(25,64),"Collect the parcel and return it to camp.",11,MUTED)
    text(d,(935,62),f"TIMELINE  {index:02d}",10,MUTED,True,"ra")
    for i, scene in enumerate(scenes):
        x = 22 + (3-len(scenes))*156 + i*312
        accent = ACCENTS[i]
        d.rounded_rectangle((x,91,x+292,496),radius=6,fill="#e9ddc7",outline="#d8c9ad")
        d.rectangle((x+14,103,x+18,117),fill=accent)
        fit(d,(x+25,101),scene["title"],252,13,True)
        fit(d,(x+14,121),scene["subtitle"],266,9,False,MUTED)
        state,step,held = map_view(im,x,scene,trace["geometry"],index,accent)
        d.rounded_rectangle((x+11,401,x+281,482),radius=4,fill="#fbf4e5")
        label, message = event_message(scene,state,step,held)
        fit(d,(x+22,412),label,248,12,True,accent)
        for j,line in enumerate(textwrap.wrap(message,43)):
            text(d,(x+22,432+j*12),line,9,MUTED)
        steps = scene["steps"][:index]
        passes = sum(s["events"]["crossed_wall"] for s in steps)
        blocked = sum(s["events"]["blocked_dash"] for s in steps)
        text(d,(x+22,465),f"TICK {state['tick']:02d}   /   PASSES {passes}   /   BLOCKS {blocked}",8,INK,True)
        if held:
            text(d,(x+278,389),"HELD AT END",7,MUTED,True,"ra")
    demo = trace["kind"] == "reference_demo"
    footer = "REFERENCE DEMO  /  scripted actions, not learned policies" if demo else "RECORDED EPISODES  /  individual examples, not mean performance"
    text(d,(24,511),footer,9,MUTED,True)
    text(d,(937,511),"Observer view  /  discrete states",9,MUTED,False,"ra")
    return im.resize((W*SCALE,H*SCALE),Image.Resampling.NEAREST)


def export(trace_path, output, *, video=True, step_seconds=.65, event_seconds=1.7, fps=10):
    raw = Path(trace_path).read_bytes()
    trace = json.loads(raw)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    n = max(len(s["steps"]) for s in trace["scenes"])
    if min(step_seconds,event_seconds,fps) <= 0:
        raise ValueError("timing values must be positive")
    schedule = []
    for index in range(n+1):
        events = [s["steps"][index-1]["events"] for s in trace["scenes"] if 0 < index <= len(s["steps"])]
        special = any(any(e.get(k) for k in ("crossed_wall","blocked_dash","picked_up","delivered")) for e in events)
        seconds = 2.5 if index in (0,n) else event_seconds if special else step_seconds
        count = max(1,round(seconds*fps))
        schedule.append({"trace_index":index,"frame_count":count,"duration_seconds":count/fps})
    interesting = next((i for i in range(1,n+1) if any(i <= len(s["steps"]) and
        s["steps"][i-1]["events"]["crossed_wall"] for s in trace["scenes"])),n)
    render_frame(trace,interesting).save(output/"spatial_wall_pass_preview.png")
    render_frame(trace,n).save(output/"spatial_wall_pass_routes.png")
    if video:
        cmd = ["ffmpeg","-y","-loglevel","error","-f","rawvideo","-vcodec","rawvideo",
               "-pix_fmt","rgb24","-s",f"{W*SCALE}x{H*SCALE}","-r",str(fps),"-i","-",
               "-an","-c:v","libx264","-preset","fast","-crf","18","-pix_fmt","yuv420p",
               "-movflags","+faststart",str(output/"spatial_wall_pass.mp4")]
        process = subprocess.Popen(cmd,stdin=subprocess.PIPE)
        try:
            for segment in schedule:
                frame = render_frame(trace,segment["trace_index"]).tobytes()
                for _ in range(segment["frame_count"]):
                    process.stdin.write(frame)
        finally:
            process.stdin.close()
            code = process.wait()
        if code:
            raise RuntimeError(f"ffmpeg failed with exit status {code}")
    (output/"timeline.json").write_text(json.dumps({"trace_sha256":hashlib.sha256(raw).hexdigest(),
        "fps":fps,"segments":schedule,"interpolation":"none; exact recorded states held between steps"},indent=2)+"\n")
    return {"output":str(output),"duration_seconds":sum(s["duration_seconds"] for s in schedule),"kind":trace["kind"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--images-only",action="store_true")
    p.add_argument("--step-seconds",type=float,default=.65)
    p.add_argument("--event-seconds",type=float,default=1.7)
    args = p.parse_args()
    print(json.dumps(export(args.trace,args.output,video=not args.images_only,
                            step_seconds=args.step_seconds,event_seconds=args.event_seconds)))


if __name__ == "__main__":
    main()
