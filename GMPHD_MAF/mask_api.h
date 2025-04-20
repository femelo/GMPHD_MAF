/**************************************************************************
* Microsoft COCO Toolbox.      version 2.0
* Data, paper, and tutorials available at:  http://mscoco.org/
* Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
* Licensed under the Simplified BSD License [see coco/license.txt]
**************************************************************************/
//https://github.com/philferriere/cocoapi/blob/master/common/maskApi.h
#pragma once

#include <cstddef>
#include <stdlib.h>
#include <math.h>


typedef unsigned int t_uint;
typedef unsigned long t_siz;
typedef unsigned char t_byte;
typedef double* t_BB;
typedef struct { t_siz h, w, m; t_uint *cnts; } t_RLE;

/* Initialize/destroy t_RLE. */
void rleInit(t_RLE *R, t_siz h, t_siz w, t_siz m, t_uint *cnts);
void rleFree(t_RLE *R);

/* Initialize/destroy t_RLE array. */
void rlesInit(t_RLE **R, t_siz n);
void rlesFree(t_RLE **R, t_siz n);

/* Encode binary masks using t_RLE. */
void rleEncode(t_RLE *R, const t_byte *mask, t_siz h, t_siz w, t_siz n);

/* Decode binary masks encoded via t_RLE. */
void rleDecode(const t_RLE *R, t_byte *mask, t_siz n);

/* Compute union or intersection of encoded masks. */
void rleMerge(const t_RLE *R, t_RLE *M, t_siz n, int intersect);

/* Compute area of encoded masks. */
void rleArea(const t_RLE *R, t_siz n, t_uint *a);

/* Compute intersection over union between masks. */
void rleIou(t_RLE *dt, t_RLE *gt, t_siz m, t_siz n, t_byte *iscrowd, double *o);

/* Compute non-maximum suppression between bounding masks */
void rleNms(t_RLE *dt, t_siz n, t_uint *keep, double thr);

/* Compute intersection over union between bounding boxes. */
void bbIou(t_BB dt, t_BB gt, t_siz m, t_siz n, t_byte *iscrowd, double *o);

/* Compute non-maximum suppression between bounding boxes */
void bbNms(t_BB dt, t_siz n, t_uint *keep, double thr);

/* Get bounding boxes surrounding encoded masks. */
void rleToBbox(const t_RLE *R, t_BB bb, t_siz n);

/* Convert bounding boxes to encoded masks. */
void rleFrBbox(t_RLE *R, const t_BB bb, t_siz h, t_siz w, t_siz n);

/* Convert polygon to encoded mask. */
void rleFrPoly(t_RLE *R, const double *xy, t_siz k, t_siz h, t_siz w);

/* Get compressed string representation of encoded mask. */
char* rleToString(const t_RLE *R);

/* Convert from compressed string representation of encoded mask. */
void rleFrString(t_RLE *R, char *s, t_siz h, t_siz w);

void rleFrStringW(t_RLE *R, wchar_t *s, t_siz h, t_siz w);
