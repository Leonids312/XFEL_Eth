"""Trace the two *slope lines* of a single DIC stripe on native RGB data.

No microscope-specific hue or polarity is assumed. A measured area refers
to the projected footprint between the two resolved slope lines, not depth.
"""
from __future__ import annotations

import cv2
import numpy as np


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _noise(a):
    med = np.median(a, axis=0)
    return np.median(np.abs(a - med), axis=0) * 1.4826


def _peaks(a, max_peaks=4):
    # Local positive maxima; retain several alternatives for path tracking.
    local = (a >= np.roll(a, -1, axis=1)) & (a > np.roll(a, 1, axis=1))
    local[:, [0, -1]] = False
    scores = np.where(local, a, -1e4)
    idx = np.argpartition(scores, -max_peaks, axis=1)[:, -max_peaks:]
    return idx, np.take_along_axis(scores, idx, axis=1)


def _follow_weak_flanks(profile, xband, lx, rx, support, pitch, sensitivity):
    """Hysteresis: a strong pair seeds a narrow search for fainter extensions.

    Searching a few neighboring locations on already identified flanks avoids
    treating the largest two noise peaks in a whole cross section as edges.
    Only observed extrema can extend the trace; do not extrapolate across
    unseen gaps or let an isolated contaminant steer the next cross section.
    """
    n = len(lx)
    strong = _runs(support)
    for l, h in sorted(strong, key=lambda r:r[1]-r[0], reverse=True):
        if h-l < max(8, int(12/pitch)):
            break
        for start, step in ((l, -1), (h-1, 1)):
            xt = np.array([lx[start], rx[start]], dtype=float)
            lo, hi = max(0, start-6), min(n, start+7)
            local = np.flatnonzero(support[lo:hi]) + lo
            ref_width = max(3., float(np.median(rx[local] - lx[local])))
            for j in range(start+step, -1 if step < 0 else n, step):
                if support[j]:
                    break
                # Small physical shifts between successive sampled lines.
                rn = 2 + int(min(2., pitch / 2.))
                x = np.rint(xt).astype(int)[:,None] + np.arange(-rn, rn+1)
                k = x - xband[j,0]
                if np.min(k) < 1 or np.max(k) >= profile.shape[1]-1:
                    break
                # Sign is tied to the learned RGB polarity, not to the side
                # that happened to be brighter at this particular section.
                v = profile[j,k] * np.array([[-1.], [1.]])
                scores = v - .23*np.abs(x-xt[:,None])
                maxima = (v >= profile[j,k-1]*np.array([[-1.],[1.]])) & (v >= profile[j,k+1]*np.array([[-1.],[1.]]))
                scores[~maxima] = -1.e4
                at = np.argmax(scores,axis=1)
                if np.min(scores[np.arange(2),at]) < .49*sensitivity:
                    break
                new = x[np.arange(2),at]
                if new[1]-new[0] < 2 or new[1]-new[0] > max(8., ref_width*1.7):
                    break
                lx[j], rx[j] = new
                xt = .45*new + .55*xt
                support[j] = True
    return support


def _pooled_ends(profile, xband, lx, rx, support, pitch, sensitivity, gap_px):
    """Two-scale, registered longitudinal averaging at the ends of a strong track.

    Each scale has its own noise measured on untouched outer sidebands. Two
    agreeing *signed* extrema are needed to extend the segment. A cap is
    certified only if both disappear into a further strip of quiet background.
    """
    n, m = profile.shape
    good = sorted(_runs(support), key=lambda v:v[1]-v[0], reverse=True)
    if not good:
        return set()
    l, h = good[0]
    span = np.arange(l,h)
    width = float(np.median(rx[span]-lx[span]))
    if width < 3 or len(span)*pitch < max(40., width*6):
        return set()
    # Median of local center estimates, resistant to occasional inclusions.
    blocks = np.array_split(span, min(12,max(2,len(span)//8)))
    pos = np.array([np.mean(i) for i in blocks])
    mid = np.array([np.median((lx[i]+rx[i])*.5) for i in blocks])
    if pos[-1] == pos[0]:
        return set()
    slope, intercept = np.polyfit(pos,mid,1)
    dy = np.arange(n)
    bend = np.interp(dy,pos,mid-(intercept+slope*pos))
    center = np.rint(intercept+slope*dy+bend).astype(int)
    r = int(np.ceil(max(12,2.5*width+3)))
    offset = np.arange(-r,r+1)
    at = center[:,None] + offset - xband[:, :1]
    if r >= m//2 or np.any(at[span,0] < 0) or np.any(at[span,-1] >= m):
        return set()
    aligned = np.take_along_axis(profile,at.clip(0,m-1),axis=1)
    # Sample separate local background windows ~1.5-2.5 widths away; remove
    # residual broad color/illumination changes before estimating tip signals.
    side = (np.abs(offset) >= r*.6)
    lwin,rwin = side & (offset<0), side & (offset>0)
    a,b = np.median(aligned[:,lwin],axis=1),np.median(aligned[:,rwin],axis=1)
    t=((offset-np.mean(offset[lwin]))/(np.mean(offset[rwin])-np.mean(offset[lwin]))).astype(np.float32)
    aligned -= a[:,None]+(b-a)[:,None]*t
    # Limit the influence of a saturated dust grain on neighboring sections.
    filtered=[]
    sigma = max(1., min(23., max(3.,width*1.6), n*pitch*.009) / pitch)
    for sy in (sigma, sigma*1.8):
        q=cv2.GaussianBlur(np.clip(aligned,-10.,10.),(1,0),1.,sigmaY=sy)
        side_noise = max(.15,float(_noise(q[::4,side].reshape(-1))))
        q=q/side_noise
        lmask=(offset>=-max(4.,width*1.35)) & (offset<=max(1.,width*.60))
        rmask=(offset<= max(4.,width*1.35)) & (offset>=-max(1.,width*.60))
        li=np.argmin(q[:,lmask],axis=1)+np.flatnonzero(lmask)[0]
        ri=np.argmax(q[:,rmask],axis=1)+np.flatnonzero(rmask)[0]
        lv,rv=-q[dy,li],q[dy,ri]
        # Extrema must be actual ridges, not the ends of the search window.
        ok=(q[dy,li]<=q[dy,li-1]) & (q[dy,li]<=q[dy,li+1]) & (q[dy,ri]>=q[dy,ri-1]) & (q[dy,ri]>=q[dy,ri+1])
        filtered.append((li,ri,lv,rv,ok))
    li,ri,sl,sr,ok=filtered[0]
    wl,wr,bl,br,ok2=filtered[1]
    agreement=(np.abs(wl-li)<=2) & (np.abs(wr-ri)<=2)
    paired=ok & ok2 & agreement & (np.minimum(sl,sr)>=sensitivity*.85) & (np.minimum(bl,br)>=sensitivity)
    paired &= (ri-li>=2) & (ri-li<=max(6.,width*1.6)) & (at[:,0]>=0) & (at[:,-1]<m)
    # Confirm the paired response in less-smoothed data to avoid spanning
    # substantial gaps with the longer filter.
    raw_l=-aligned[dy,li]; raw_r=aligned[dy,ri]
    check=cv2.blur(np.column_stack((raw_l, raw_r)),(1,max(1,int(round(max(5.,gap_px)/pitch)))))
    paired &= (np.min(check,axis=1) > .5)
    cert=set()
    for start,step in ((l,-1),(h-1,1)):
        tip=start
        for j in range(start+step,-1 if step<0 else n,step):
            if not (paired[j] or support[j]):
                break
            if paired[j] and not support[j]:
                if max(abs(center[j]+offset[li[j]]-lx[tip]),abs(center[j]+offset[ri[j]]-rx[tip]))>max(4.,width*.40):
                    break
                lx[j],rx[j]=center[j]+offset[li[j]],center[j]+offset[ri[j]]
                support[j]=True
            tip=j
        # A noise maximum a few pixels beyond a tip is not continuing
        # damage. Conversely do not certify a cap if either *coherent* flank
        # persists at either scale for another longitudinal resolution element.
        extent=max(4,int(np.ceil(4*sigma)))
        outside=tip+step*np.arange(1,extent*2+1)
        inside=tip-step*np.arange(0,extent*2)
        if min(outside.min(),inside.min())<0 or max(outside.max(),inside.max())>=n:
            continue
        if np.any(~support[inside[:extent]]) or not np.all(at[outside[extent:],0]>=0) or not np.all(at[outside[extent:],-1]<m):
            continue
        quiet=outside[extent:]
        # Check each flank in a narrow tube about its terminal coordinate,
        # not the maximum of a wide noisy search band.
        vals=[]
        for terminal,sign in ((lx[tip],-1),(rx[tip],1)):
            pos0=np.clip(terminal-center[tip]+r+np.arange(-2,3),0,2*r)
            signal=np.max(sign*aligned[:,pos0],axis=1)
            bg=np.max(sign*aligned[:,(pos0 + (r if pos0.mean()<r else -r)).clip(0,2*r)],axis=1)
            # The maximum over a tube has a positive bias; use a control
            # tube with the same width, and the same smoothing, to remove it.
            med=np.median(bg); sc=max(.18,float(_noise(bg)))
            vals.append((signal-med)/sc)
        vals=np.array(vals)
        aligned_pair = paired[quiet] & (np.abs(center[quiet] + offset[li[quiet]]-lx[tip]) < max(4.,width*.40)) & (np.abs(center[quiet] + offset[ri[quiet]]-rx[tip]) < max(4.,width*.40))
        if (np.min(np.median(vals[:,inside],axis=1)) >= sensitivity*.60 and
            np.max(np.median(vals[:,quiet],axis=1)) < sensitivity*.50 and
            np.max(np.mean(vals[:,quiet] > sensitivity,axis=1)) < .2 and
            np.mean(aligned_pair) < .15):
            cert.add((tip,step))
    return cert


def detect_dic(roi_rgb, *, axis="vertical", gap_px=6, sensitivity=3.0):
    """Return a closed contour only when two supported edges and ends exist.

    axis describes the stripe's long axis, not the shear direction. gap_px is
    in native pixels along the long axis. No results are shared between ROIs.
    """
    if axis not in ("vertical", "horizontal"):
        raise ValueError("axis must be vertical or horizontal")
    img = np.asarray(roi_rgb)
    if img.ndim == 2:
        img = img[:, :, None]
    if img.ndim != 3 or img.shape[2] not in (1, 3, 4):
        raise ValueError("expected grayscale or RGB image")
    if not np.isfinite(sensitivity) or sensitivity <= 0 or not np.isfinite(gap_px) or gap_px < 0:
        raise ValueError("sensitivity must be positive and gap_px nonnegative")
    if axis == "horizontal":
        img = img.transpose(1, 0, 2)
    length, width = img.shape[:2]
    diagnostics = {}

    def result(status, reason, *, contour=None, paths=None):
        return {"status": status, "contour": contour,
                "diagnostics": {**diagnostics, "reason": reason}, "paths": paths or []}

    if min(length, width) < 16:
        return result("待复核", "点位过小，无法分开两侧坡线与背景")
    # Bin along the stripe, never across its width. All reported coordinates
    # are mapped back to the native image including the bin-center offset.
    n = max(16, min(length, 1000, 1_200_000 // width))
    pitch = length / n
    yy = (np.arange(n) + .5) * pitch - .5
    # Convert one native channel at a time. A 16-bit TIFF retains its weak
    # increments; a full-frame float RGB copy is unnecessary even for 4K images.
    planes, reliability = [], []
    retained = []
    integer = np.issubdtype(img.dtype, np.integer)
    if integer:
        limits = np.iinfo(img.dtype)
    for channel in range(min(3,img.shape[2])):
        plane = img[...,channel]
        if integer:
            valid = ((plane != limits.min) & (plane != limits.max)).astype(np.uint8)
            if valid.mean() < .01:
                continue
            if valid.min() == 0:
                rel = cv2.resize(valid.astype(np.float32), (width,n), interpolation=cv2.INTER_AREA)
                rel = cv2.GaussianBlur(rel, (0,0), .9, sigmaY=max(.8,min(9.,max(1.1,length*.003))/pitch))
                rel[rel < .25] = 0.
            else:
                rel = None
            reliability.append(rel)
        planes.append(cv2.resize(plane.astype(np.float32), (width,n), interpolation=cv2.INTER_AREA))
        retained.append(channel)
    if not planes:
        return result("待复核", "所有通道均大范围截顶或截底")
    diagnostics["channels"] = "".join("RGB"[i] for i in retained) if img.shape[2] > 1 else "灰度"
    a = np.stack(planes,axis=2)
    del planes
    quality = None
    if any(v is not None for v in reliability):
        quality = np.stack([np.ones((n,width),np.float32) if v is None else v for v in reliability],axis=2)
    del reliability
    if not np.all(np.isfinite(a)):
        return result("待复核", "非有限像素，无法评估局部背景")
    # Blur in a narrow neighborhood. Filtering is only for localization; the
    # original channels are not stretched, gamma-corrected, or converted to 8 bit.
    smooth = cv2.GaussianBlur(a, (0, 0), .9, sigmaY=max(.8, min(9., max(1.1, length*.003)) / pitch))
    residual = smooth - cv2.GaussianBlur(smooth, (0, 1), max(5., min(32., width / 8)))
    if smooth.ndim == 2:
        smooth, residual = smooth[..., None], residual[..., None]
    c = smooth.shape[2]
    sample = residual[::max(1, n // 90), ::2].reshape(-1, c)
    # A floor prevents a mostly clipped or quantized channel getting infinite weight.
    noise = np.maximum(_noise(sample), .004 * np.percentile(np.abs(sample), 99.7, axis=0))
    if np.issubdtype(img.dtype, np.integer):
        noise = np.maximum(noise, .65)
    noise = np.maximum(noise, max(1e-8, float(np.max(np.abs(smooth))) * 1.e-7))
    z = residual / noise
    if quality is not None:
        z *= quality
    # Vote for a persistent, near-axial feature, not simply the darkest spot.
    rows = np.arange(0, n, max(1, n // 160))
    v = np.minimum(np.max(np.abs(z[rows]), axis=2), 2 * sensitivity + 1)
    v = cv2.GaussianBlur(v, (0, 1), 1.4)
    xx = np.arange(width)
    unit_y = (np.arange(n) + .5) / n - .5
    best = (-np.inf, 0, 0)
    for drift in np.linspace(-min(width * .7, length * .22), min(width * .7, length * .22), 19):
        x = xx[None, :] + np.rint(drift * unit_y[rows]).astype(int)[:, None]
        track = np.take_along_axis(v, x.clip(0, width - 1), axis=1)
        track[(x < 2) | (x >= width - 2)] = 0
        votes = track.mean(axis=0)
        # Weak center prior; arbitrary point order or stripe polarity does not enter.
        votes -= .18 * ((xx - width / 2) / (width / 2)) ** 2
        k = int(np.argmax(votes))
        if votes[k] > best[0]:
            best = votes[k], k, drift
    center = best[1] + best[2] * unit_y
    # Sidebands far outside the provisional edges estimate an independent local
    # background: high-pass halos of a *single* bright stripe must not count
    # as its second edge.
    radius = max(6, min(int(width * .24), 120))
    offsets = np.arange(-radius, radius + 1)
    xband = np.rint(center[:, None]).astype(int) + offsets
    safe = xband.clip(0, width - 1)
    samples = smooth[np.arange(n)[:, None], safe]
    distance = max(3, int(width * .065))
    xa = np.clip(safe[:, :1] - np.arange(1, distance + 1), 0, width - 1)
    xb = np.clip(safe[:, -1:] + np.arange(1, distance + 1), 0, width - 1)
    left = np.median(smooth[np.arange(n)[:, None], xa], axis=1)
    right = np.median(smooth[np.arange(n)[:, None], xb], axis=1)
    level = ((safe - xa.mean(axis=1)[:, None]) /
             np.maximum(1., (xb.mean(axis=1) - xa.mean(axis=1))[:, None])).astype(np.float32)
    base = left[:, None] + (right - left)[:, None] * level[:, :, None]
    normed = (samples - base) / noise
    if quality is not None:
        normed *= quality[np.arange(n)[:,None], safe]
    # Learn the signed RGB slope response of this ROI. For each row compare
    # a few possible separations; two supported flanks beat a one-sided spike.
    scores = np.zeros(n)
    vectors = np.zeros((n, c))
    rn = np.arange(n)
    for d in np.unique(np.rint(np.geomspace(2, max(3, radius), 12)).astype(int)):
        u, w = normed[:, :-d], normed[:, d:]
        delta = w - u
        mag = np.sqrt(np.sum(delta * delta, axis=2) + 1.e-7)
        uproj = -np.sum(u * delta, axis=2) / mag
        strength = np.minimum(uproj, mag - uproj)
        strength[(xband[:, :-d] < 1) | (xband[:, d:] >= width - 1)] = -np.inf
        j = strength.argmax(axis=1)
        s = strength[rn, j]
        take = s > scores
        scores[take] = s[take]
        vectors[take] = (delta[rn, j] / mag[rn, j, None])[take]
    good = scores >= max(sensitivity, np.percentile(scores, 64))
    if good.sum() < max(4, n * .015):
        return result("未检出", "未找到连续的双边颜色响应")
    direction = vectors[good].mean(axis=0)
    if np.linalg.norm(direction) < .24:
        return result("待复核", "候选坡沿的亮暗极性不一致")
    direction /= np.linalg.norm(direction)
    profile = (normed @ direction).astype(np.float32)
    # Re-estimate scalar noise after projection, allowing for RGB covariance.
    scalar_noise = max(.65, float(_noise((sample / noise) @ direction)))
    profile /= scalar_noise
    profile[(xband < 1) | (xband >= width - 1)] = 0.
    li, ls = _peaks(-profile)
    ri, rs = _peaks(profile)
    k = li.shape[1]
    ll = np.repeat(li, k, axis=1)
    rr = np.tile(ri, (1, k))
    lval, rval = np.repeat(ls, k, axis=1), np.tile(rs, (1, k))
    strength = np.minimum(lval, rval)
    feasible = (rr > ll + 1) & (rr - ll <= radius) & (strength > 0)
    value = np.where(feasible, np.minimum(strength, 7) +
                     .15 * np.minimum(np.maximum(lval, rval), 8), -8.)
    # Viterbi tracking in the native transverse coordinates. A dummy state
    # lets the track stop in genuine background instead of forcing a pair.
    ll = np.take_along_axis(safe, ll, axis=1)
    rr = np.take_along_axis(safe, rr, axis=1)
    value -= .45 * np.abs((ll + rr) / 2 - center[:, None]) / radius
    value -= max(2., sensitivity)
    states = k * k
    back = np.zeros((n, states + 1), dtype=np.uint8)
    prev = np.zeros(states + 1, dtype=float)
    for i in range(n):
        trans = np.zeros((states + 1, states + 1))
        if i:
            trans[:-1, :-1] = -.7 * (
                np.abs(ll[i-1, :, None] - ll[i, None]) +
                np.abs(rr[i-1, :, None] - rr[i, None])) / max(1., pitch)
        trans[-1, :-1] = -2.
        trans[:-1, -1] = -2.
        trans += prev[:, None]
        back[i] = np.argmax(trans, axis=0)
        prev = trans[back[i], np.arange(states + 1)] + np.r_[value[i], 0.]
    picks = np.empty(n, dtype=int)
    picks[-1] = int(np.argmax(prev))
    for i in range(n - 1, 0, -1):
        picks[i-1] = back[i, picks[i]]
    jj = np.minimum(picks, states - 1)
    lx, rx = ll[rn, jj], rr[rn, jj]
    supported = (picks < states) & feasible[rn, jj] & (strength[rn, jj] >= sensitivity * .72)
    # Bridge gaps below the *strong* seed threshold with fresh flank evidence
    # first; the separate gap limit below applies to genuinely missing evidence.
    supported = _follow_weak_flanks(profile, xband, lx, rx, supported, pitch, sensitivity)
    certified_caps = _pooled_ends(profile, xband, lx, rx, supported, pitch, sensitivity, gap_px)
    # Interpolate only short gaps if endpoints agree in position. Never bridge
    # a large occluded region just because a line exists on its other side.
    connected = supported.copy()
    for lo, hi in _runs(~supported):
        if lo and hi < n and (hi-lo) * pitch <= gap_px and max(
                abs(lx[lo-1]-lx[hi]), abs(rx[lo-1]-rx[hi])) <= max(3, gap_px):
            connected[lo:hi] = True
    spans = sorted(_runs(connected), key=lambda r: r[1]-r[0], reverse=True)
    minimum = max(8, int(np.ceil(max(12., length * .025) / pitch)))
    spans = [s for s in spans if s[1]-s[0] >= minimum]
    if not spans:
        return result("未检出", "未找到足够长且同向的双边")
    lo, hi = spans[0]
    edge_rows = np.arange(lo, hi)
    good_rows = edge_rows[supported[edge_rows]]
    xa = np.interp(edge_rows, good_rows, lx[good_rows])
    xb = np.interp(edge_rows, good_rows, rx[good_rows])
    ya = yy[edge_rows]
    def coords(x, y):
        x, y = np.rint(x).astype(int), np.rint(y).astype(int)
        return np.column_stack((x, y) if axis == "vertical" else (y, x)).tolist()
    paths = [coords(xa, ya), coords(xb, ya)]
    local_left = np.clip(lx - xband[:,0],0,profile.shape[1]-1)
    local_right = np.clip(rx - xband[:,0],0,profile.shape[1]-1)
    measured_snr = np.minimum(-profile[rn,local_left], profile[rn,local_right])
    diagnostics.update(paired_rows=int(len(good_rows)), sampled_rows=int(n),
                       width_px=round(float(np.median(xb-xa)), 2),
                       median_snr=round(float(np.median(measured_snr[good_rows])), 2))
    # Test all credible segments, not just the strongest; a large obstruction
    # can separate an otherwise edge-clipped track into several pieces.
    if any(i <= 1 or j >= n - 1 for i, j in spans):
        return result("图像截断", "双边延伸到长轴方向的图像边缘", paths=paths)
    if len(spans) > 1 and spans[1][1]-spans[1][0] > max(minimum, (hi-lo) * .12):
        return result("待复核", "存在不连续的双边段，未跨过缺口封闭", paths=paths)
    if np.any(xa < 2) or np.any(xb > width - 3):
        return result("待复核", "坡沿超出点位侧边", paths=paths)
    width_m = max(2., float(np.percentile(xb-xa, 70)))
    if (hi-lo) * pitch < width_m * 2.0:
        return result("待复核", "候选形状不像细长条纹", paths=paths)
    # A vanished weak flank is not an end-cap. Accept only a taper at each
    # end or a paired, abrupt termination into independently quiet background.
    for first in (True, False):
        ii = lo if first else hi - 1
        if (ii, -1 if first else 1) in certified_caps:
            continue
        edge = slice(0, min(3,len(xa))) if first else slice(-3,None)
        tip_width = float(np.median((xb-xa)[edge]))
        ext = np.arange(ii-1, max(-1, ii-1-max(4, int(14/pitch))), -1) if first else np.arange(ii+1, min(n,ii+1+max(4,int(14/pitch))))
        if ext.size < 2:
            return result("待复核", "缺少端点外背景", paths=paths)
        # Check known flank positions outside the proposed cap, with lateral
        # tolerance. A persistent opposite flank precludes auto-closure.
        flank = []
        for xt, sign in ((float(np.median(xa[edge])), -1), (float(np.median(xb[edge])), 1)):
            ix = np.rint(xt - xband[ext,0]).astype(int)
            loc = (ix[:,None] + np.arange(-2,3)).clip(0, profile.shape[1]-1)
            flank.append(np.max(sign * profile[ext[:,None],loc],axis=1))
        outside = np.maximum(*flank)
        if np.mean(outside >= sensitivity) > .48:
            return result("待复核", "端点外仍有单侧坡线，未强制封口", paths=paths)
        taper = tip_width <= max(3., width_m*.75)
        abrupt = (np.median(measured_snr[good_rows[:3] if first else good_rows[-3:]]) >= sensitivity*1.5 and
                  np.median(outside) < sensitivity*.75)
        if not (taper or abrupt):
            return result("待复核", "端点既未明显收窄，也无清晰的双边终止", paths=paths)
    contour = paths[0] + paths[1][::-1]
    return result("自动", "连续双边且端点可辨", contour=contour)
