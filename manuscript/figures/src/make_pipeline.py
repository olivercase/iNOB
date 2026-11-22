# Figure 1 of the iNOB paper: the pipeline, as editable SVG at Nature size (120 mm wide, 1.5 column).
# Coordinates are points (viewBox 0 0 340 H), so font-size 7 is 7 pt. Palette from house/figures/nature_palette.json.
import os, json
HERE=os.path.dirname(os.path.abspath(__file__)); FIG=os.path.dirname(HERE)
STYLE=next(d for d in (os.path.join(HERE,'..','..','..','house','figures'), os.path.join(HERE,'style')) if os.path.exists(os.path.join(d,'nature_palette.json')))
NAT=json.load(open(os.path.join(STYLE,'nature_palette.json')))['palette']
TXT='#000000'; LINE=NAT['grey'][5]
W=340; BW=150; BH=26; GAP=16
LX=10; RX=W-10-BW          # left and right column x
CL=LX+BW/2; CR=RX+BW/2    # column centres
o=[]; a=o.append
def box(x,y,w,h,num,title,sub,fill,stroke,sw=0.75,id_=None):
    a(f'<g id="{id_}"><rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
      f'<text x="{x+w/2}" y="{y+11}" text-anchor="middle" font-weight="700">{num}  {title}</text>'
      f'<text x="{x+w/2}" y="{y+20.5}" text-anchor="middle">{sub}</text></g>')
def arrow(d,id_=None):
    a(f'<path d="{d}" fill="none" stroke="{LINE}" stroke-width="0.75" marker-end="url(#ah)"/>')
stages=[(1,'Modality','MXG, EXG or both','mxg/exg'),(2,'Anatomy','tissue surfaces for the target region','surfaces'),
        (3,'Conductivities','one value per tissue','sigma'),(4,'FEM meshing','multi-tissue tetrahedral volume','volume'),
        (5,'Sensor array','magnetometers and/or surface electrodes',None)]
Y0=8; ys=[Y0+i*(BH+GAP) for i in range(5)]
GREY=NAT['grey'][0]; 
for (n,t,s,_),y in zip(stages,ys): box(LX,y,BW,BH,n,t,s,GREY,LINE,id_=f'stage-{n}')
# edges between model-building stages
for i,(n,t,s,e) in enumerate(stages[:-1]):
    y1=ys[i]+BH; y2=ys[i+1]
    arrow(f'M{CL},{y1} V{y2}')
    a(f'<text x="{CL+5}" y="{(y1+y2)/2+2.5}">{e}</text>')
box(RX,ys[4],BW,BH,6,'Sources','dipole location, orientation, strength',NAT['orange'][0],LINE,id_='stage-6')
# forward solve
fy=ys[4]+BH+34; fw=BW; fx=(W-fw)/2
box(fx,fy,fw,BH,7,'Forward solve','DUNEuro finite-element leadfield',NAT['blue'][1],LINE,id_='stage-7')
mid=ys[4]+BH+17
xl=fx+fw/2-8; xr=fx+fw/2+8
arrow(f'M{CL},{ys[4]+BH} V{mid} H{xl} V{fy}')
arrow(f'M{CR},{ys[4]+BH} V{mid} H{xr} V{fy}')
a(f'<text x="{(CL+xl)/2}" y="{mid-3}" text-anchor="middle">sensors</text><text x="{(CR+xr)/2}" y="{mid-3}" text-anchor="middle">dipoles</text>')
# goal
gy=fy+BH+GAP
box(fx,gy,fw,BH,8,'Trials to detect','amplitude, SNR and averages needed',NAT['blue'][1],NAT['blue'][5],sw=1.5,id_='stage-8-goal')
arrow(f'M{W/2},{fy+BH} V{gy}')
a(f'<text x="{W/2+5}" y="{fy+BH+GAP/2+2.5}">leadfield</text>')
H=gy+BH+8
svg=(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H:.0f}" width="120mm" height="{H*120/W:.1f}mm" font-family="Helvetica, Arial, sans-serif" font-size="7" fill="{TXT}">'
     f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker></defs>'
     f'<rect width="{W}" height="{H:.0f}" fill="#fff"/>'+''.join(o)+'</svg>')
open(os.path.join(FIG,'fig1_pipeline.svg'),'w').write(svg); print('H',H)
import sys; sys.path.insert(0, STYLE)
import nature_style as ns; ns.svg_to_pdf(os.path.join(FIG,'fig1_pipeline.svg'))
