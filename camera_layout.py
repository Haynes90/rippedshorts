"""Conservative camera-tile detection and aspect-preserving composition.

Unknown regions remain visible. Face positions alone never define tile borders.
Participant IDs are spatial tracks, not claims about a person's identity.
"""
from __future__ import annotations

import math
from statistics import median


def separators(frame, axis):
    import numpy as np
    gray = frame.mean(axis=2) if frame.ndim == 3 else frame.astype(float)
    differences = np.max(np.abs(np.diff(frame.astype(float),axis=axis)),axis=2) if frame.ndim==3 else np.abs(np.diff(gray.astype(float),axis=axis))
    # Require a real sustained line, not a face, jaw movement or background object.
    other = 1-axis
    contrast = np.mean(differences, axis=other)
    coverage = np.mean(differences > 16, axis=other)
    length = gray.shape[axis]
    candidates = [i+1 for i in range(len(contrast)) if .18*length < i < .82*length and contrast[i] > 12 and coverage[i] > .65]
    groups = []
    for i in candidates:
        if groups and i-groups[-1][-1] < 8:
            groups[-1].append(i)
        else:
            groups.append([i])
    return [int(median(g)) for g in groups]


def detect_regions(frame, faces):
    """Require visible separators plus at least two actual participant regions."""
    height, width = frame.shape[:2]
    xs, ys = separators(frame, 1), separators(frame, 0)
    if len(xs) > 1 or len(ys) > 1:
        return None  # busy graphics or an unknown many-window layout
    boxes = []
    xedges, yedges = [0]+xs+[width], [0]+ys+[height]
    for top,bottom in zip(yedges,yedges[1:]):
        for left,right in zip(xedges,xedges[1:]):
            if right-left < width*.18 or bottom-top < height*.18:
                return None
            occupants = [f for f in faces if left <= f[0]+f[2]/2 < right and top <= f[1]+f[3]/2 < bottom]
            boxes.append({'id': str(len(boxes)), 'rect': [left,top,right-left,bottom-top],
                          'kind': 'camera' if len(occupants)==1 else 'decoration' if float(frame[top:bottom,left:right].std(axis=(0,1)).max())<2 else 'context', 'faces': [list(map(int,f)) for f in occupants]})
    cameras = [r for r in boxes if r['kind']=='camera']
    if not xs and not ys:
        return {'layout':'single' if len(faces)==1 else 'unknown','regions':boxes,'confidence':0.0}
    if len(cameras)<2:
        return None
    return {'layout':'mixed' if len(cameras)!=len(boxes) else 'grid' if xs and ys else 'side_by_side' if xs else 'stacked',
            'regions':boxes, 'confidence':.8, 'basis':'sustained_visual_separators_and_faces'}


def stable_regions(observations, width, height):
    valid = [o for o in observations if o and o['confidence']>=.8]
    if len(valid) < 3 or len(valid)<len(observations)*.75:
        return None
    first = valid[0]
    if any(o['layout']!=first['layout'] or len(o['regions'])!=len(first['regions']) for o in valid):
        return None
    for a in valid:
        for r,s in zip(first['regions'],a['regions']):
            if r['kind']!=s['kind'] or any(abs(x-y)>max(width,height)*.015 for x,y in zip(r['rect'],s['rect'])):
                return None
    return first


def speaker_sections(samples, energy, duration):
    """Mixed audio gates motion; only correlated, distinctive mouth motion selects a tile."""
    import numpy as np
    sections = [(0.,None)]
    current,pending,count,last = None,None,0,0.
    for index,(timestamp,motions) in enumerate(samples):
        if index<5:
            continue
        window=samples[max(0,index-7):index+1]
        audio=[energy[min(len(energy)-1,max(0,int(t*20)))] for t,_ in window] if energy else []
        scores=[]
        for tile in motions:
            mouth=[m.get(tile,0) for _,m in window]
            correlation=float(np.corrcoef(mouth,audio)[0,1]) if audio and np.std(audio)>.0001 and np.std(mouth)>.002 else 0.
            if not math.isfinite(correlation):
                correlation=0.
            if correlation>.45 and median(mouth)>.008:
                scores.append((correlation*median(mouth),tile))
        scores.sort(reverse=True)
        desired=scores[0][1] if scores and (len(scores)==1 or scores[0][0]>1.6*scores[1][0]) else None
        if desired==current:
            pending,count=None,0
            continue
        pending,count=(desired,count+1) if desired==pending else (desired,1)
        if count>=3 and timestamp-last>=3:
            current=desired;last=timestamp
            sections.append((timestamp,current))
    return sections


def composition_graph(regions, primary=None, label='0:v', suffix='0'):
    """Preserve complete camera rectangles with letterboxing, never face-only crops."""
    cameras=[r for r in regions if r['kind']=='camera']
    context=[r for r in regions if r['kind']=='context']
    if context or not cameras:
        return f'[{label}]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p[v{suffix}]'
    cameras=sorted(cameras,key=lambda r: (r['id']!=primary,r['id'])) if primary is not None else cameras
    n=len(cameras)
    if n==2:
        panels=[(1080,1280,0,0),(1080,640,0,1280)] if primary is not None else [(1080,960,0,0),(1080,960,0,960)]
    elif n==3:
        panels=[(1080,1080,0,0),(540,840,0,1080),(540,840,540,1080)]
    elif n==4 and primary is not None:
        # Keep the active tile legible. Others remain together in a stable overview,
        # rather than distorting three tiny tiles to fill narrow panels.
        panels=[(1080,1080,0,0),(360,840,0,1080),(360,840,360,1080),(360,840,720,1080)]
    elif n==4:
        panels=[(540,960,0,0),(540,960,540,0),(540,960,0,960),(540,960,540,960)]
    else:
        return None
    graph=[f'[{label}]split={n}'+''.join(f'[tile{suffix}_{i}]' for i in range(n))]
    for i,(camera,(w,h,x,y)) in enumerate(zip(cameras,panels)):
        left,top,cw,ch=camera['rect']
        # Inset one pixel, then make even: neighboring borders cannot bleed in.
        left,top=left+2,top+2
        cw,ch=max(2,(cw-4)//2*2),max(2,(ch-4)//2*2)
        graph.append(f'[tile{suffix}_{i}]crop={cw}:{ch}:{left}:{top},scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p[p{suffix}_{i}]')
    graph.append(''.join(f'[p{suffix}_{i}]' for i in range(n))+f'xstack=inputs={n}:layout='+ '|'.join(f'{x}_{y}' for _,_,x,y in panels)+f'[v{suffix}]')
    return ';'.join(graph)


def analyze(video, start, duration, cascade):
    import cv2
    import numpy as np
    capture=cv2.VideoCapture(str(video))
    observations=[];frames=[]
    try:
        for fraction in (.05,.25,.5,.75,.95):
            capture.set(cv2.CAP_PROP_POS_MSEC,(start+duration*fraction)*1000)
            ok,frame=capture.read()
            if not ok:
                return None
            frame=cv2.resize(frame,(960,540))
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=list(cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30)))
            observations.append(detect_regions(frame,faces));frames.append(frame)
        layout=stable_regions(observations,960,540)
        if not layout:
            return None
        samples=[];previous={}
        for t in np.arange(0,duration,.5):
            capture.set(cv2.CAP_PROP_POS_MSEC,(start+float(t))*1000)
            ok,frame=capture.read()
            if not ok:
                break
            frame=cv2.resize(frame,(960,540));gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            motions={}
            for region in layout['regions']:
                if region['kind']!='camera':
                    continue
                x,y,w,h=region['rect']
                faces=list(cascade.detectMultiScale(gray[y:y+h,x:x+w],scaleFactor=1.1,minNeighbors=5,minSize=(30,30)))
                if len(faces)!=1:
                    previous.pop(region['id'],None)
                    continue
                fx,fy,fw,fh=faces[0]
                mouth=cv2.resize(gray[y+fy+fh//2:y+fy+fh,x+fx:x+fx+fw],(64,32))
                if region['id'] in previous:
                    motions[region['id']]=float(np.mean(cv2.absdiff(previous[region['id']],mouth)))/255
                previous[region['id']]=mouth
            samples.append((float(t),motions))
        return layout,samples
    finally:
        capture.release()


def build(video,start,duration,width,height,cascade,energy):
    result=analyze(video,start,duration,cascade)
    if not result:
        return None
    layout,samples=result
    for region in layout['regions']:
        x,y,w,h=region['rect']
        region['rect']=[int(x*width/960),int(y*height/540),int(w*width/960),int(h*height/540)]
    sections=speaker_sections(samples,energy,duration)
    graphs=[];outputs=[]
    for i,(t,primary) in enumerate(sections):
        end=sections[i+1][0] if i+1<len(sections) else duration
        if primary is not None and len(layout['regions'])>=4:
            # If secondary faces would be unreadable, retain the larger stable grid.
            primary=None
        graphs.append(f'[0:v]trim=start={t:.3f}:end={end:.3f},setpts=PTS-STARTPTS[camera{i}]')
        graph=composition_graph(layout['regions'],primary,label=f'camera{i}',suffix=f'c{i}')
        if not graph:
            return None
        graphs.append(graph);outputs.append(f'[vc{i}]')
    graphs.append(''.join(outputs)+f'concat=n={len(outputs)}:v=1:a=0[v]')
    return ';'.join(graphs),{'layout':layout,'sections':sections,'basis':'visual_tiles_and_correlated_mouth_motion','recipient_identity_inferred':False}
