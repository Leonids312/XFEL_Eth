"""Edge-supported closed contours of mixed bright/dark damage pits, one cell at a time."""

from __future__ import annotations

import cv2
import numpy as np


def _result(status, reason, contour=None, **diagnostics):
    return {"status": status, "contour": contour, "diagnostics": {"reason": reason, **diagnostics}}


def _kernel(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)


def _features(rgb):
    """Subtract a robust illumination surface, keeping positive AND negative colour deviations."""
    h, w = rgb.shape[:2]
    y, x = np.mgrid[-1:1:complex(h), -1:1:complex(w)].astype(np.float32)
    model = np.stack((np.ones_like(x), x, y, x*x, x*y, y*y), axis=-1)
    step = max(1, min(h, w) // 46)
    m = model[::step, ::step].reshape(-1, 6)
    v = rgb[::step, ::step].reshape(len(m), -1)
    zero = np.median(v,axis=0)
    v = v-zero
    good = np.ones(len(m), bool)
    for _ in range(3):
        coeff = np.linalg.lstsq(m[good], v[good], rcond=None)[0]
        diff = v - m @ coeff
        mad = np.median(np.abs(diff-np.median(diff, axis=0)), axis=0) * 1.483
        # Relative to the dynamic range; works with byte, word and floating-point TIFF.
        floor = np.maximum(np.ptp(v, axis=0) * 0.001, np.finfo(np.float32).eps)
        scale = np.maximum(mad, floor)
        good = np.max(np.abs(diff) / scale, axis=1) < 3.0
        if good.sum() < max(10, len(m)//3):
            break
    z = (rgb-zero - model @ coeff) / scale
    # A fixed luminance projection can cancel chromatic rims; use the measured
    # background colour covariance, regularized to tolerate identical channels.
    bg = (diff/scale)[good]
    if len(bg)>=15 and z.shape[-1]>1:
        cov = np.cov(bg.T)
        eigen, axes = np.linalg.eigh(cov)
        floor_eig = max(.12*float(eigen.max()), .12)
        transform = (axes*(1/np.sqrt(np.maximum(eigen,floor_eig)))) @ axes.T
        z = z @ transform.astype(np.float32)
    return np.ascontiguousarray(np.clip(z, -10000, 10000), dtype=np.float32)


def _sample_ring(contour):
    p = contour[:, 0, :].astype(np.float32)
    closed = np.concatenate((p, p[:1]))
    arclength = np.r_[0, np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    n = max(16, min(1600, int(arclength[-1]/1.15)))
    s = np.arange(n) * arclength[-1] / n
    points = np.stack([np.interp(s, arclength, closed[:, j]) for j in (0, 1)], axis=-1)
    rad = max(2, min(7, round(n*.012)))
    points = sum((rad+1-abs(i))*np.roll(points, i, axis=0)
                 for i in range(-rad,rad+1)) / (rad+1)**2
    tangent = np.roll(points, -2, axis=0)-np.roll(points, 2, axis=0)
    tangent /= np.maximum(1e-6, np.linalg.norm(tangent, axis=1, keepdims=True))
    orient = np.sign(cv2.contourArea(contour, oriented=True))
    normals = np.stack((tangent[:, 1], -tangent[:, 0]), axis=-1) * orient
    return points.astype(np.float32), normals.astype(np.float32)


def _pick_path(cost):
    """Closed narrow-band shortest path, anchoring at the best-defined arc."""
    anchor = np.argmax(np.max(cost, axis=1)-np.median(cost, axis=1))
    cost = np.roll(cost, -anchor, axis=0)
    n, m = cost.shape
    fixed = np.argmax(cost[0])
    dp = np.full(m, -1e9, dtype=np.float32)
    dp[fixed] = cost[0, fixed]
    prev = np.zeros((n,m), dtype=np.int8)
    ix = np.arange(m)
    links = np.clip(ix[None, :] + np.arange(-2,3)[:,None], 0, m-1)
    penalty = 0.31 * np.abs(links - ix)
    for i in range(1,n):
        links_cost = dp[links] - penalty
        best = np.argmax(links_cost, axis=0)
        prev[i] = links[best, ix] - ix
        dp = cost[i] + links_cost[best, ix]
    dp[np.abs(ix-fixed)>2] = -1e9
    path = np.empty(n, dtype=int)
    path[-1] = dp.argmax()
    for i in range(n-1, 0, -1):
        path[i-1] = path[i] + prev[i,path[i]]
    return np.roll(path, anchor)


def _radial_band(cnt, area):
    """For compact pits, construct one ordered set of rays (no folded search normals)."""
    m = cv2.moments(cnt)
    center = np.array([m["m10"]/m["m00"],m["m01"]/m["m00"]],np.float32)
    if cv2.pointPolygonTest(cnt, tuple(float(x) for x in center), False) <= 0:
        return _sample_ring(cnt)
    box = cv2.boundingRect(cnt)
    x,y,w,h = box
    mask = np.zeros((h+4,w+4),np.uint8)
    origin = np.array((x-2,y-2),np.int32)
    cv2.drawContours(mask,[cnt-origin],0,1,cv2.FILLED)
    n = min(1000,max(28,round(2*np.sqrt(np.pi*area)/1.15)))
    theta = np.arange(n)*2*np.pi/n
    rays = np.stack((np.cos(theta),np.sin(theta)),axis=1).astype(np.float32)
    rs = np.arange(np.hypot(w,h)+2,dtype=np.float32)
    coords = (center-origin).astype(np.float32) + rays[:,None,:]*rs[None,:,None]
    profile = cv2.remap(mask,coords[...,0],coords[...,1],cv2.INTER_NEAREST)
    if np.mean(np.any(np.diff(profile.astype(int), axis=1)>0,axis=1))>.12:
        return _sample_ring(cnt)  # strongly concave, so retain the native polygon geometry
    rim = np.max(rs * profile,axis=1)
    width = max(2,round(n*.030))
    rim = np.median(np.stack([np.roll(rim,i) for i in range(-width,width+1)]),axis=0)
    rim = sum((width+1-abs(i))*np.roll(rim,i) for i in range(-width,width+1))/(width+1)**2
    return center+rays*rim[:,None],rays


def _runs_without_support(ok):
    """Return index sequences including their two neighbouring measured endpoints."""
    if np.all(ok):
        return []
    good = np.flatnonzero(ok)
    if not len(good):
        return [np.arange(len(ok)+1) % len(ok)]
    return [np.arange(a,b+1) % len(ok) for a,b in
            zip(good,np.r_[good[1:],good[0]+len(ok)]) if b-a>1]


def _coherent_arcs(z, contour, normals, span, threshold):
    """Matched signed transition along a short arc, calibrated on shifted null traces.

    A faint RGB step can be invisible at a single vertex, but visible on several
    equivalent neighbouring normal profiles. The null traces use the same
    geometry, interpolation, smoothing and channel correlations as the pit.
    """
    h,w = z.shape[:2]
    # Do not average in a guessed direction across long unsupported intervals.
    n = max(1, min(4, round(len(contour)*.009)))
    shifts = [(0.,0.), (.21,.18), (-.21,.18), (.21,-.18), (-.21,-.18),
              (.22,0), (-.22,0), (0,.23), (0,-.23)]
    translate = np.array(shifts,dtype=np.float32) * [w,h]
    def reads(s):
        loc = (contour[:,None,:]+(normals*s)[:,None,:]+translate[None,:,:]).astype(np.float32)
        c = cv2.remap(z,loc[...,0],loc[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        return c[...,None] if c.ndim==2 else c
    delta = reads(-span) - reads(span)
    local_mean = sum(np.roll(delta,k,axis=0) for k in range(-n,n+1))/(2*n+1)
    amplitude = np.sqrt(np.mean(local_mean**2,axis=2))
    center = contour[:,None,:]+translate[None,1:,:]
    valid = ((center[...,0]>span)&(center[...,0]<w-span)&
             (center[...,1]>span)&(center[...,1]<h-span))
    null = amplitude[:,1:][valid]
    if len(null)<30:
        return np.zeros(len(contour),bool)
    med = np.median(null)
    cut = max(.3, med+threshold*1.483*np.median(np.abs(null-med)))
    local = delta[:,0,:]
    coherent = np.sum(local*local_mean[:,0,:],axis=1)>0
    return ((amplitude[:,0]>=cut) & coherent &
            (np.sqrt(np.mean(local**2,axis=1)) >= .6*cut))


def _native_refine(native, contour, normals, factor, supported):
    """For large cells, re-localize measured arcs on the full-depth original pixels.

    The search is limited to one coarse pixel either side; unsupported sections
    are not promoted to measurements. Only sparse samples from the raw ROI are read.
    """
    h,w = native.shape[:2]
    position = (contour+.5)*factor-.5
    normal = normals*factor
    normal = normal/np.maximum(.01,np.linalg.norm(normal,axis=1,keepdims=True))
    tangent = np.stack((-normal[:,1],normal[:,0]),axis=-1)
    radius = min(10,max(1,round(float(factor.max()))))
    offsets = np.arange(-radius,radius+1,dtype=np.float32)
    coords = position[:,None,:]+normal[:,None,:]*offsets[None,:,None]
    def sample(shift):
        total = 0.
        for shift_t in (-1.,0.,1.):
            p = coords+normal[:,None,:]*shift+tangent[:,None,:]*shift_t
            # NumPy gather avoids copying/casting the entire native TIFF.
            xx = np.clip(np.rint(p[...,0]).astype(int),0,w-1)
            yy = np.clip(np.rint(p[...,1]).astype(int),0,h-1)
            total += native[yy,xx,:3].astype(np.float32)
        return total/3
    contrast = np.linalg.norm(sample(-1.25)-sample(1.25),axis=-1)
    strength = contrast/np.maximum(contrast.max(axis=1,keepdims=True),1e-7)
    selected = _pick_path(strength-.15*np.abs(offsets)[None,:]/radius)
    delta = offsets[selected]*supported
    return (position+normal*delta[:,None]+.5)/factor-.5


def detect_pit(roi_rgb, *, gap_px=6, sensitivity=3.0):
    """Return a cell-local, nonconvex polygon or an explicit reason for abstaining.

    A local two-sided colour anomaly only locates candidate pits. Their final
    boundaries follow colour transition arcs, including arcs of opposite signs.
    A short evidence gap may be joined; long unsupported closures are not measurements.
    """
    src = np.asarray(roi_rgb)
    if src.ndim == 2:
        src = src[..., None]
    if src.ndim != 3 or src.shape[2] not in (1, 3, 4) or min(src.shape[:2]) < 14:
        return _result("待复核", "坑位图像太小或通道不支持")
    height, width = src.shape[:2]
    scale = min(1., 1100./max(height, width))
    native = src
    if scale < 1:
        if min(height,width)*scale<14:
            return _result("待复核","分区过于狭长，不能可靠识别紧凑损伤坑")
        src = cv2.resize(src[..., :3], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if src.ndim == 2:
            src = src[...,None]
    src = np.ascontiguousarray(src[..., :3], dtype=np.float32)
    if not np.all(np.isfinite(src)):
        return _result("待复核", "图像包含无效像素")
    src = cv2.GaussianBlur(src, (0,0), 1.3)
    if src.ndim == 2:
        src = src[...,None]
    h, w = src.shape[:2]
    factor = np.array([width/w, height/h], dtype=np.float32)
    z = _features(src)
    signal = np.sqrt(np.mean(z*z, axis=-1))
    threshold = max(1.5, float(sensitivity))
    mask = np.array(signal >= threshold, dtype=np.uint8)
    # Reject thin scratches. Close small separations of dark and light material
    # to locate a seed; this mask is never used as the final damage boundary.
    first_radius = max(1, min(4, round(min(h,w)*.006)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(first_radius))
    # A diagonal scratch can cross a pit and run all the way to the cell border.
    # Estimate pit-core thickness independently of component length, then break
    # thin branches with a scale-proportional opening before tracing the rim.
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    center = dist[h//10:h-h//10, w//10:w-w//10]
    # Region-scale seeding (independent of maximum *measured-edge* gap).
    # Closing first unites the two halves separated by a zero-contrast waist.
    radius = max(first_radius, min(22, round(float(center.max())*.50)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(radius))
    core_radius = min(42, max(first_radius, round(float(center.max())*.57)))
    if core_radius > first_radius:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(core_radius))
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ranked = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < max(12, first_radius**2*3) or area > .65*h*w:
            continue
        x,y,cw,ch = cv2.boundingRect(cnt)
        m = cv2.moments(cnt)
        dx = (m["m10"]/m["m00"] - .5*w)/w
        dy = (m["m01"]/m["m00"] - .5*h)/h
        if max(abs(dx),abs(dy))>.38 and not (x<=0 or y<=0 or x+cw>=w or y+ch>=h):
            continue
        center_prior = np.exp(-6*(dx*dx+dy*dy))
        rw,rh = cv2.minAreaRect(cnt)[1]
        if min(rw,rh) < max(rw,rh)*.12 or min(rw,rh) < 2:
            continue
        solidity = area/max(1.,cv2.contourArea(cv2.convexHull(cnt)))
        if solidity < .34:
            continue
        touches = x<=0 or y<=0 or x+cw>=w or y+ch>=h
        score = area * center_prior**2 * solidity**2 * (0.32 if touches else 1)
        ranked.append((score,cnt,area,touches))
    if not ranked:
        return _result("未检出", "未找到区别于背景的紧凑坑位")
    ranked.sort(key=lambda item: item[0], reverse=True)
    score,cnt,area,touches = ranked[0]
    if touches:
        return _result("图像截断","候选坑与当前分区边缘相接")
    if len(ranked)>1 and ranked[1][0]>.64*score:
        return _result("待复核","同一坑位有多个相近候选轮廓")
    points, normals = _radial_band(cnt,area)
    limit = min(24, max(3, round(np.sqrt(area/np.pi)*.23)))
    offsets = np.arange(-limit,limit+1, dtype=np.float32)
    pts = points[:,None,:] + normals[:,None,:]*offsets[None,:,None]
    smooth = cv2.GaussianBlur(z, (0,0), min(2.4,max(1.3, limit*.3)))
    if smooth.ndim == 2:
        smooth = smooth[...,None]
    def sample(shift, data=z):
        loc = pts + normals[:,None,:]*np.float32(shift)
        colors = cv2.remap(data, loc[...,0], loc[...,1], cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        return colors[...,None] if colors.ndim==2 else colors
    span = min(5., max(2., .065*np.sqrt(area/np.pi)))
    inner, outer = sample(-span), sample(span)
    change = np.sqrt(np.mean((inner-outer)**2, axis=2))
    fine = np.sqrt(np.mean((sample(-.85)-sample(.85))**2,axis=2))
    fall = np.sqrt(np.mean(inner*inner,axis=2)) - np.sqrt(np.mean(outer*outer,axis=2))
    coarse_delta = sample(-span,smooth)-sample(span,smooth)
    coarse = np.sqrt(np.mean(coarse_delta**2,axis=-1))
    # A fine-scale derivative localizes the transition even on a virtually
    # noiseless calibration target. Do not clip the gradient to a flat plateau.
    # If the trial point lies on an inner dark/bright boundary, the area
    # immediately OUTSIDE it usually still differs strongly from the local
    # background. Prefer an outer material/background boundary when available.
    further = sample(2*span)
    outer_strength = np.sqrt(np.mean(further*further, axis=2))
    reward = (.48*np.log1p(change)+.40*np.log1p(coarse)+.63*np.log1p(fine)
              + .34*np.tanh(fall*.5)
              - .36*np.log1p(np.minimum(outer_strength,20))
              - .045*np.abs(offsets)[None,:])
    # Estimate noise from flat-region colour *differences*, not the spread of
    # colours over the entire field (which penalises slow brightness drift).
    stride = int(round(2*span))
    def background_cut(data):
        diff_x = np.sqrt(np.mean((data[:,stride:,:]-data[:,:-stride,:])**2,axis=2))
        diff_y = np.sqrt(np.mean((data[stride:,:,:]-data[:-stride,:,:])**2,axis=2))
        sampled = np.r_[diff_x[::5,::5].ravel(), diff_y[::5,::5].ravel()]
        median = np.median(sampled)
        return max(.40, median+threshold*1.483*np.median(np.abs(sampled-median)))
    cut = background_cut(z)
    coarse_cut = background_cut(smooth)
    # A gradual edge may be weak over one aperture but persists at twice the
    # scale. Require evidence at both scales and the same colour direction.
    wide_delta = sample(-2*span)-further
    wide = np.sqrt(np.mean(wide_delta**2,axis=-1))
    concordant = np.sum((inner-outer)*wide_delta,axis=-1)>0
    supported = ((change >= cut) | (coarse>=coarse_cut) |
                 ((change >= .55*cut) & (wide>=1.4*cut) & concordant))
    # Search for an actually measured route between arcs before bridging any
    # gap. Constraining consecutive offsets prevents jumping to isolated specks.
    loc = _pick_path(reward + .95*supported)
    ix = np.arange(len(loc))
    contour = pts[ix,loc]
    ok = supported[ix,loc]
    ok |= _coherent_arcs(z, contour, normals, span, threshold)
    if scale < 1:
        contour = _native_refine(native, contour, normals, factor, ok)
    coverage = float(ok.mean())
    gaps = _runs_without_support(ok)
    lengths = [float(np.linalg.norm(np.diff(contour[g]*factor,axis=0),axis=1).sum())
               for g in gaps]
    max_gap = max(lengths,default=0.)
    diagnostics = {"edge_coverage":round(coverage,3), "max_gap_px":round(max_gap,2)}
    perimeter = float(cv2.arcLength((contour*factor).astype(np.float32), True))
    m = cv2.moments(cnt)
    center_seed = (m["m10"]/m["m00"],m["m01"]/m["m00"])
    open_arc = (cv2.pointPolygonTest(cnt, center_seed, False)<=0 and
                area < .82*cv2.contourArea(cv2.convexHull(cnt)))
    if max_gap > min(float(gap_px), .075*perimeter)+1e-4 or coverage < .83 or open_arc:
        reason = "候选像未闭合的弧带，不能把弧带本身当作坑面积" if open_arc else "坑沿存在超限缺口或有效边段不足"
        result = _result("待复核",reason,**diagnostics)
        # Preserve separate measured runs; never show the rest as a measured
        # contour, and never publish its area without explicit manual review.
        px = np.rint((contour+.5)*factor-.5).astype(np.int32)
        result["paths"] = [cv2.approxPolyDP(px[g[1:-1],None,:],.55,False)[:,0,:].tolist()
                           for g in _runs_without_support(~ok) if len(g)>3]
        result["review_contour"] = cv2.approxPolyDP(px[:,None,:], .65, True)[:,0,:].tolist()
        return result
    for group in gaps:
        contour[group] = np.linspace(contour[group[0]],contour[group[-1]],len(group))
    if (np.any(contour<.5) or np.any(contour[:,0]>w-1.5)
            or np.any(contour[:,1]>h-1.5)):
        return _result("图像截断","轮廓接触当前分区边缘",**diagnostics)
    pixel = np.rint((contour+.5)*factor-.5).astype(np.int32)
    poly = cv2.approxPolyDP(pixel[:,None,:], .65, True)
    if len(poly)<3 or cv2.contourArea(poly)<8:
        return _result("待复核","闭合轮廓面积过小",**diagnostics)
    result = _result("自动","局部颜色边缘拼接闭合",poly[:,0,:].tolist(),**diagnostics)
    result["bridges"] = [[pixel[g[0]].tolist(),pixel[g[-1]].tolist()] for g in gaps]
    return result
