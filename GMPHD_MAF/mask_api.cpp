/**************************************************************************
* Microsoft COCO Toolbox.      version 2.0
* Data, paper, and tutorials available at:  http://mscoco.org/
* Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
* Licensed under the Simplified BSD License [see coco/license.txt]
**************************************************************************/
// https://github.com/philferriere/cocoapi/blob/master/common/maskApi.c
#include "mask_api.h"

t_uint umin(t_uint a, t_uint b) { return (a < b) ? a : b; }
t_uint umax(t_uint a, t_uint b) { return (a > b) ? a : b; }

void rleInit(t_RLE *R, t_siz h, t_siz w, t_siz m, t_uint *cnts) {
	R->h = h; R->w = w; R->m = m; R->cnts = (m == 0) ? 0 : (t_uint*)malloc(sizeof(t_uint)*m);
	t_siz j; if (cnts) for (j = 0; j < m; j++) R->cnts[j] = cnts[j];
}

void rleFree(t_RLE *R) {
	free(R->cnts); R->cnts = 0;
}

void rlesInit(t_RLE **R, t_siz n) {
	t_siz i; *R = (t_RLE*)malloc(sizeof(t_RLE)*n);
	for (i = 0; i < n; i++) rleInit((*R) + i, 0, 0, 0, 0);
}

void rlesFree(t_RLE **R, t_siz n) {
	t_siz i; for (i = 0; i < n; i++) rleFree((*R) + i); free(*R); *R = 0;
}

void rleEncode(t_RLE *R, const t_byte *M, t_siz h, t_siz w, t_siz n) {
	t_siz i, j, k, a = w * h; t_uint c, *cnts; t_byte p;
	cnts = (t_uint*)malloc(sizeof(t_uint)*(a + 1));
	for (i = 0; i < n; i++) {
		const t_byte *T = M + a * i; k = 0; p = 0; c = 0;
		for (j = 0; j < a; j++) { if (T[j] != p) { cnts[k++] = c; c = 0; p = T[j]; } c++; }
		cnts[k++] = c; rleInit(R + i, h, w, k, cnts);
	}
	free(cnts);
}

void rleDecode(const t_RLE *R, t_byte *M, t_siz n) {
	t_siz i, j, k; for (i = 0; i < n; i++) {
		t_byte v = 0; for (j = 0; j < R[i].m; j++) {
			for (k = 0; k < R[i].cnts[j]; k++) *(M++) = v; v = !v;
		}
	}
}

void rleMerge(const t_RLE *R, t_RLE *M, t_siz n, int intersect) {
	t_uint *cnts, c, ca, cb, cc, ct; int v, va, vb, vp;
	t_siz i, a, b, h = R[0].h, w = R[0].w, m = R[0].m; t_RLE A, B;
	if (n == 0) { rleInit(M, 0, 0, 0, 0); return; }
	if (n == 1) { rleInit(M, h, w, m, R[0].cnts); return; }
	cnts = (t_uint*)malloc(sizeof(t_uint)*(h*w + 1));
	for (a = 0; a < m; a++) cnts[a] = R[0].cnts[a];
	for (i = 1; i < n; i++) {
		B = R[i]; if (B.h != h || B.w != w) { h = w = m = 0; break; }
		rleInit(&A, h, w, m, cnts); ca = A.cnts[0]; cb = B.cnts[0];
		v = va = vb = 0; m = 0; a = b = 1; cc = 0; ct = 1;
		while (ct > 0) {
			c = umin(ca, cb); cc += c; ct = 0;
			ca -= c; if (!ca && a < A.m) { ca = A.cnts[a++]; va = !va; } ct += ca;
			cb -= c; if (!cb && b < B.m) { cb = B.cnts[b++]; vb = !vb; } ct += cb;
			vp = v; if (intersect) v = va && vb; else v = va || vb;
			if (v != vp || ct == 0) { cnts[m++] = cc; cc = 0; }
		}
		rleFree(&A);
	}
	rleInit(M, h, w, m, cnts); free(cnts);
}

void rleArea(const t_RLE *R, t_siz n, t_uint *a) {
	t_siz i, j; for (i = 0; i < n; i++) {
		a[i] = 0; for (j = 1; j < R[i].m; j += 2) a[i] += R[i].cnts[j];
	}
}

void rleIou(t_RLE *dt, t_RLE *gt, t_siz m, t_siz n, t_byte *iscrowd, double *o) {
	t_siz g, d; t_BB db, gb; int crowd;
	db = (double*)malloc(sizeof(double)*m * 4); rleToBbox(dt, db, m);
	gb = (double*)malloc(sizeof(double)*n * 4); rleToBbox(gt, gb, n);
	bbIou(db, gb, m, n, iscrowd, o); free(db); free(gb);
	for (g = 0; g < n; g++) for (d = 0; d < m; d++) if (o[g*m + d] > 0) {
		crowd = iscrowd != NULL && iscrowd[g];
		if (dt[d].h != gt[g].h || dt[d].w != gt[g].w) { o[g*m + d] = -1; continue; }
		t_siz ka, kb, a, b; t_uint c, ca, cb, ct, i, u; int va, vb;
		ca = dt[d].cnts[0]; ka = dt[d].m; va = vb = 0;
		cb = gt[g].cnts[0]; kb = gt[g].m; a = b = 1; i = u = 0; ct = 1;
		while (ct > 0) {
			c = umin(ca, cb); if (va || vb) { u += c; if (va&&vb) i += c; } ct = 0;
			ca -= c; if (!ca && a < ka) { ca = dt[d].cnts[a++]; va = !va; } ct += ca;
			cb -= c; if (!cb && b < kb) { cb = gt[g].cnts[b++]; vb = !vb; } ct += cb;
		}
		if (i == 0) u = 1; else if (crowd) rleArea(dt + d, 1, &u);
		o[g*m + d] = (double)i / (double)u;
	}
}

void rleNms(t_RLE *dt, t_siz n, t_uint *keep, double thr) {
	t_siz i, j; double u;
	for (i = 0; i < n; i++) keep[i] = 1;
	for (i = 0; i < n; i++) if (keep[i]) {
		for (j = i + 1; j < n; j++) if (keep[j]) {
			rleIou(dt + i, dt + j, 1, 1, 0, &u);
			if (u > thr) keep[j] = 0;
		}
	}
}

void bbIou(t_BB dt, t_BB gt, t_siz m, t_siz n, t_byte *iscrowd, double *o) {
	double h, w, i, u, ga, da; t_siz g, d; int crowd;
	for (g = 0; g < n; g++) {
		t_BB G = gt + g * 4; ga = G[2] * G[3]; crowd = iscrowd != NULL && iscrowd[g];
		for (d = 0; d < m; d++) {
			t_BB D = dt + d * 4; da = D[2] * D[3]; o[g*m + d] = 0;
			w = fmin(D[2] + D[0], G[2] + G[0]) - fmax(D[0], G[0]); if (w <= 0) continue;
			h = fmin(D[3] + D[1], G[3] + G[1]) - fmax(D[1], G[1]); if (h <= 0) continue;
			i = w * h; u = crowd ? da : da + ga - i; o[g*m + d] = i / u;
		}
	}
}

void bbNms(t_BB dt, t_siz n, t_uint *keep, double thr) {
	t_siz i, j; double u;
	for (i = 0; i < n; i++) keep[i] = 1;
	for (i = 0; i < n; i++) if (keep[i]) {
		for (j = i + 1; j < n; j++) if (keep[j]) {
			bbIou(dt + i * 4, dt + j * 4, 1, 1, 0, &u);
			if (u > thr) keep[j] = 0;
		}
	}
}

void rleToBbox(const t_RLE *R, t_BB bb, t_siz n) {
	t_siz i; for (i = 0; i < n; i++) {
		t_uint h, w, x, y, xs, ys, xe, ye, xp, cc, t; t_siz j, m;
		h = (t_uint)R[i].h; w = (t_uint)R[i].w; m = R[i].m;
		m = ((t_siz)(m / 2)) * 2; xs = w; ys = h; xe = ye = 0; cc = 0;
		if (m == 0) { bb[4 * i + 0] = bb[4 * i + 1] = bb[4 * i + 2] = bb[4 * i + 3] = 0; continue; }
		for (j = 0; j < m; j++) {
			cc += R[i].cnts[j]; t = cc - j % 2; y = t % h; x = (t - y) / h;
			if (j % 2 == 0) xp = x; else if (xp < x) { ys = 0; ye = h - 1; }
			xs = umin(xs, x); xe = umax(xe, x); ys = umin(ys, y); ye = umax(ye, y);
		}
		bb[4 * i + 0] = xs; bb[4 * i + 2] = xe - xs + 1;
		bb[4 * i + 1] = ys; bb[4 * i + 3] = ye - ys + 1;
	}
}

void rleFrBbox(t_RLE *R, const t_BB bb, t_siz h, t_siz w, t_siz n) {
	t_siz i; for (i = 0; i < n; i++) {
		double xs = bb[4 * i + 0], xe = xs + bb[4 * i + 2];
		double ys = bb[4 * i + 1], ye = ys + bb[4 * i + 3];
		double xy[8] = { xs,ys,xs,ye,xe,ye,xe,ys };
		rleFrPoly(R + i, xy, 4, h, w);
	}
}

int t_uintCompare(const void *a, const void *b) {
	t_uint c = *((t_uint*)a), d = *((t_uint*)b); return c > d ? 1 : c < d ? -1 : 0;
}

void rleFrPoly(t_RLE *R, const double *xy, t_siz k, t_siz h, t_siz w) {
	/* upsample and get discrete points densely along entire boundary */
	t_siz j, m = 0; double scale = 5; int *x, *y, *u, *v; t_uint *a, *b;
	x = (int*)malloc(sizeof(int)*(k + 1)); y = (int*)malloc(sizeof(int)*(k + 1));
	for (j = 0; j < k; j++) x[j] = (int)(scale*xy[j * 2 + 0] + .5); x[k] = x[0];
	for (j = 0; j < k; j++) y[j] = (int)(scale*xy[j * 2 + 1] + .5); y[k] = y[0];
	for (j = 0; j < k; j++) m += umax(abs(x[j] - x[j + 1]), abs(y[j] - y[j + 1])) + 1;
	u = (int*)malloc(sizeof(int)*m); v = (int*)malloc(sizeof(int)*m); m = 0;
	for (j = 0; j < k; j++) {
		int xs = x[j], xe = x[j + 1], ys = y[j], ye = y[j + 1], dx, dy, t, d;
		int flip; double s; dx = abs(xe - xs); dy = abs(ys - ye);
		flip = (dx >= dy && xs > xe) || (dx<dy && ys>ye);
		if (flip) { t = xs; xs = xe; xe = t; t = ys; ys = ye; ye = t; }
		s = dx >= dy ? (double)(ye - ys) / dx : (double)(xe - xs) / dy;
		if (dx >= dy) for (d = 0; d <= dx; d++) {
			t = flip ? dx - d : d; u[m] = t + xs; v[m] = (int)(ys + s * t + .5); m++;
		}
		else for (d = 0; d <= dy; d++) {
			t = flip ? dy - d : d; v[m] = t + ys; u[m] = (int)(xs + s * t + .5); m++;
		}
	}
	/* get points along y-boundary and downsample */
	free(x); free(y); k = m; m = 0; double xd, yd;
	x = (int*)malloc(sizeof(int)*k); y = (int*)malloc(sizeof(int)*k);
	for (j = 1; j < k; j++) if (u[j] != u[j - 1]) {
		xd = (double)(u[j] < u[j - 1] ? u[j] : u[j] - 1); xd = (xd + .5) / scale - .5;
		if (floor(xd) != xd || xd<0 || xd>w - 1) continue;
		yd = (double)(v[j] < v[j - 1] ? v[j] : v[j - 1]); yd = (yd + .5) / scale - .5;
		if (yd < 0) yd = 0; else if (yd > h) yd = h; yd = ceil(yd);
		x[m] = (int)xd; y[m] = (int)yd; m++;
	}
	/* compute rle encoding given y-boundary points */
	k = m; a = (t_uint*)malloc(sizeof(t_uint)*(k + 1));
	for (j = 0; j < k; j++) a[j] = (t_uint)(x[j] * (int)(h)+y[j]);
	a[k++] = (t_uint)(h*w); free(u); free(v); free(x); free(y);
	qsort(a, k, sizeof(t_uint), t_uintCompare); t_uint p = 0;
	for (j = 0; j < k; j++) { t_uint t = a[j]; a[j] -= p; p = t; }
	b = (t_uint*)malloc(sizeof(t_uint)*k); j = m = 0; b[m++] = a[j++];
	while (j < k) if (a[j] > 0) b[m++] = a[j++]; else {
		j++; if (j < k) b[m - 1] += a[j++];
	}
	rleInit(R, h, w, m, b); free(a); free(b);
}

char* rleToString(const t_RLE *R) {
	/* Similar to LEB128 but using 6 bits/char and ascii chars 48-111. */
	t_siz i, m = R->m, p = 0; long x; int more;
	char *s = (char*)malloc(sizeof(char)*m * 6);
	for (i = 0; i < m; i++) {
		x = (long)R->cnts[i]; if (i > 2) x -= (long)R->cnts[i - 2]; more = 1;
		while (more) {
			char c = x & 0x1f; x >>= 5; more = (c & 0x10) ? x != -1 : x != 0;
			if (more) c |= 0x20; c += 48; s[p++] = c;
		}
	}
	s[p] = 0; return s;
}

void rleFrString(t_RLE *R, char *s, t_siz h, t_siz w) {
	t_siz m = 0, p = 0, k; long x; int more; t_uint *cnts;
	while (s[m]) m++; cnts = (t_uint*)malloc(sizeof(t_uint)*m); m = 0;
	while (s[p]) {
		x = 0; k = 0; more = 1;
		while (more) {
			char c = s[p] - 48; x |= (c & 0x1f) << 5 * k;
			more = c & 0x20; p++; k++;
			if (!more && (c & 0x10)) x |= -1 << 5 * k;
		}
		if (m > 2) x += (long)cnts[m - 2]; cnts[m++] = (t_uint)x;
	}
	rleInit(R, h, w, m, cnts); free(cnts);
}
void rleFrStringW(t_RLE *R, wchar_t *s, t_siz h, t_siz w) {
	t_siz m = 0, p = 0, k; long x; int more; t_uint *cnts;
	while (s[m]) m++; cnts = (t_uint*)malloc(sizeof(t_uint)*m); m = 0;
	while (s[p]) {
		x = 0; k = 0; more = 1;
		while (more) {
			wchar_t c = s[p] - 48; x |= (c & 0x1f) << 5 * k;
			more = c & 0x20; p++; k++;
			if (!more && (c & 0x10)) x |= -1 << 5 * k;
		}
		if (m > 2) x += (long)cnts[m - 2]; cnts[m++] = (t_uint)x;
	}
	rleInit(R, h, w, m, cnts); free(cnts);
}
