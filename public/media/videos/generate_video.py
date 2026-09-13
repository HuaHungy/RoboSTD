from PIL import Image, ImageDraw, ImageFont
import cv2, os, math, numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
W,H,FPS=1280,720,30
font_path='C:/Windows/Fonts/arial.ttf'
bold_path='C:/Windows/Fonts/arialbd.ttf'
def font(sz,b=False):
    try:return ImageFont.truetype(bold_path if b else font_path,sz)
    except:return ImageFont.load_default()
WHITE=(248,250,252); NAVY=(13,27,42); BLUE=(46,125,210); GREEN=(41,160,103); RED=(214,83,80); ORANGE=(232,145,42); GRAY=(110,120,130)
writer=cv2.VideoWriter(str(OUT/'robostd_project_video.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W,H))
def frame_bg(c=NAVY):
    return Image.new('RGB',(W,H),c)
def centered(draw,text,y,ft,fill=WHITE):
    box=draw.textbbox((0,0),text,font=ft); draw.text(((W-(box[2]-box[0]))//2,y),text,font=ft,fill=fill)
def emit(im):
    writer.write(cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR))
def card(title,sub,dur=3):
    im=frame_bg(); d=ImageDraw.Draw(im)
    d.rectangle((0,0,W,12),fill=GREEN); centered(d,title,210,font(62,True)); centered(d,sub,300,font(30), (190,205,218))
    for t,x,col in [('SINGLE',290,BLUE),('→',615,ORANGE),('DUAL',850,GREEN)]: centered_x=d.textbbox((0,0),t,font=font(34,True))[2] if False else None; d.text((x,430),t,font=font(34,True),fill=col)
    for _ in range(int(dur*FPS)): emit(im)
def centered_x(*a): pass
def gif_frames(path):
    im=Image.open(path); out=[]
    try:
      n=im.n_frames
    except:n=1
    for i in range(n):
      im.seek(i); fr=im.convert('RGB');
      # cover crop to 16:9
      ar=fr.width/fr.height; target=W/H
      if ar>target:
        nw=int(fr.height*target); l=(fr.width-nw)//2; fr=fr.crop((l,0,l+nw,fr.height))
      else:
        nh=int(fr.width/target); t=(fr.height-nh)//2; fr=fr.crop((0,t,fr.width,t+nh))
      out.append(fr.resize((W,H),Image.Resampling.LANCZOS))
    return out
def demo(name,subtitle,path,dur=8,accent=GREEN,metric=None):
    frames=gif_frames(path); count=max(1,int(dur*FPS));
    for j in range(count):
      im=frames[j%len(frames)].copy(); d=ImageDraw.Draw(im,'RGBA')
      d.rectangle((0,0,W,92),fill=(8,18,30,225)); d.text((40,22),name,font=font(38,True),fill=WHITE); d.text((42,65),subtitle,font=font(20),fill=(205,220,230))
      d.rounded_rectangle((W-310,22,W-40,78),radius=22,fill=accent+(235,));
      if metric: d.text((W-285,36),metric,font=font(24,True),fill=WHITE)
      emit(im)
def text_card(title,lines,dur=4):
    im=frame_bg(); d=ImageDraw.Draw(im); d.rectangle((0,0,W,12),fill=ORANGE); centered(d,title,130,font(48,True));
    y=250
    for txt,col in lines:
      centered(d,txt,y,font(31,True),col); y+=65
    for _ in range(int(dur*FPS)): emit(im)
# Intro
card('RoboSTD','Zero-Shot Single-to-Dual Transfer via Sagittal Mirroring',4)
text_card('The challenge',[('Abundant single-arm demonstrations',BLUE),('do not directly provide balanced bimanual supervision',WHITE),('Action-space gap  ·  Coordination gap  ·  Arm-side bias',ORANGE)],5)
text_card('RoboSTD pipeline',[('1  Physically valid mirroring',BLUE),('2  LLM-guided spatio-temporal reconstruction',ORANGE),('3  Executable pseudo-bimanual supervision',GREEN)],5)
# demos
base=ROOT/'demos'/'realworld-demo'
demo('Bowl Placement','Rigid object · direct transfer',base/'bowl-placement.gif',8,BLUE,'40.0% → 77.3%')
demo('Towel Storage','Deformable object · direct transfer',base/'towel-storage.gif',8,GREEN,'RoboSTD')
demo('Flower Arrangement','Precision placement · direct transfer',base/'flower-arrangement.gif',8,GREEN,'RoboSTD')
text_card('Spatial coordination · Cup Collection',[('Single-arm                         2.00',BLUE),('RoboSTD w/o LLM               2.76',ORANGE),('RoboSTD                             4.22',GREEN),('Oracle                               3.92',GRAY)],5)
demo('Cup Collection','Arm assignment + conflict constraints',base/'cup-collection.gif',8,GREEN,'Average 4.22')
text_card('Long-horizon coordination · Sandwich Making',[('Completion   55%  →  63%  →  74%',GREEN),('Success          26%  →  30%  →  50%',GREEN),('Stage-level language supervision',ORANGE)],5)
demo('Sandwich Making','Stage 1 → Stage 2 → Stage 3 → Stage 4',base/'sandwich-making.gif',9,GREEN,'Success 50%')
text_card('Bias mitigation',[('Density gap                 45.8% → 25.6%',BLUE),('Arm-side bias gap         28.3 pp → 13.0 pp',GREEN),('Broader left–right coverage',WHITE)],5)
card('RoboSTD','A data-centric bridge from Single to Dual',5)
writer.release()
print(OUT/'robostd_project_video.mp4')
